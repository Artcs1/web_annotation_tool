from flask import Flask, render_template, jsonify, request, send_from_directory, session
from scipy.optimize import linear_sum_assignment

import uuid
import os
import json
import glob
import random
import numpy as np
import pickle
import cv2
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

class VideoAnnotationApp:
    """Video annotation application with MongoDB backend"""
    
    def __init__(self):
        # Flask app setup
        self.app = Flask(__name__)
        self.app.secret_key = os.getenv('SECRET_KEY')
        self.app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 181  # 180 days

        if not self.app.secret_key:
            raise ValueError("SECRET_KEY environment variable is not set! Please create a .env file with SECRET_KEY.")
        
        self.MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        self.MONGO_DB = 'video_annotations'
        self.annotator_per_clip = 3
        self.number_of_clips = 15
        self.VIDEO_BASE_PATH = 'videos'
        self.YES_NO_VIDEO_BASE_PATH = 'yes_no_videos'
        self.VIDEO_VALIDATION_BASE_PATH = 'validation_videos'
        self.GTS_VALIDATION_BASE_PATH = 'validation_gts'
        self.FRAME_EXTENSION = '.jpeg'
        self.FRAME_PADDING = 4
        self.FRAME_COUNT_PADDING = 5
        self.choices_annotatedframe = [1,21,41]
        self.global_idx = 0

        self.BASE_DIR = Path(__file__).parent
        self.VIDEOS_DIR = self.BASE_DIR / "videos"
        self.ANNOTATIONS_FILE = self.BASE_DIR / "coarse_group" / "all_annotations.json"
        self.DETECTIONS_CACHE_FILE = self.BASE_DIR / "detections_cache.json"

        with open(self.ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
            self.COARSE_ANNOTATIONS = json.load(f)

        def load_detections_cache():
            if self.DETECTIONS_CACHE_FILE.exists():
                print("[INFO] Loading cached detections from", self.DETECTIONS_CACHE_FILE)
                with open(self.DETECTIONS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print("[WARN] Detections cache not found. Run preprocess_detections.py first!")
                return {}
        
        self.DETECTIONS_CACHE = load_detections_cache()
        
        # MongoDB setup
        try:
            self.client = MongoClient(self.MONGO_URI)
            self.db = self.client[self.MONGO_DB]
            self.annotations_collection = self.db['annotations']
            self.yes_no_annotations_collection = self.db['yes_no_annotations']
            self.finegrained_annotations_collection = self.db['finegrained_annotations']
            print(f"✅ Connected to MongoDB: {self.MONGO_DB}")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None

        def generate_boxes(groups):

            boxes = []
            for i, group in enumerate(groups):
                boxes.append({
                    'id': f'box_{str(self.global_idx).zfill(5)}',
                    'tl_x1': group[0],
                    'tl_y1': group[1],
                    'br_x1': group[2],
                    'br_y1': group[3],
                    'confidence': random.randint(1, 5)  # Random confidence 1-5
                })
                self.global_idx+=1

            return boxes

        def generate_random_boxes(num_boxes=None, img_width=1920, img_height=1080):
            """Generate random bounding boxes for a frame"""
            if num_boxes is None:
                num_boxes = random.randint(1, 5)  # Random 1-5 boxes
            
            boxes = []
            for i in range(num_boxes):
                box_width = random.randint(50, int(img_width * 0.3))
                box_height = random.randint(50, int(img_height * 0.3))
                
                x = random.randint(0, img_width - box_width)
                y = random.randint(0, img_height - box_height)
                
                tl_x1 = int(x-(box_width//2))
                tl_y1 = int(y-(box_width//2))
                br_x1 = int(x+(box_width//2))
                br_y1 = int(y+(box_width//2))
                
                boxes.append({
                    'id': f'box_{i}',
                    'tl_x1': tl_x1,
                    'tl_y1': tl_y1,
                    'br_x1': br_x1,
                    'br_y1': br_y1,
                    'confidence': random.randint(1, 5)  # Random confidence 1-5
                })

            return boxes


        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
        
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            if interArea == 0:
                return 0.0
        
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            unionArea = boxAArea + boxBArea - interArea
            return interArea / unionArea
        
        def best_iou_matching(boxesA, boxesB, threshold=0.5):
            if not boxesA or not boxesB:
                return 0
        
            nA, nB = len(boxesA), len(boxesB)
            iou_matrix = np.zeros((nA, nB))
        
            for i in range(nA):
                for j in range(nB):
                    iou_matrix[i, j] = iou(boxesA[i], boxesB[j])
        
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matches = sum(iou_matrix[i, j] >= threshold for i, j in zip(row_ind, col_ind))
            return matches
        
        @self.app.route('/api/get-annotator-id')
        def get_annotator_id():
            """Get or create unique annotator ID"""
            if 'annotator_id' not in session:
                session['annotator_id'] = str(uuid.uuid4())
                session.permanent = True
            return jsonify({'annotator_id': session['annotator_id']})
         
        ################################
        ### COARSE ANNOTATION ROUTES ###
        ################################

        @self.app.route('/annotation')
        def annotation():
            return render_template('annotation.html')
        
        @self.app.route("/thank_you")
        def thank_you():
            return render_template("thank_you.html")
      
        @self.app.route('/api/detect-videos')
        def detect_videos():
            """Detect available video folders"""

            # Get completed annotations
            try:
                ann_results = self.annotations_collection.find({"annotator_id": session['annotator_id']})
            except:
                ann_results = []

            annotated_clips = [(result['globalIndex']-1, result['videoInfo']['annotationFrame']) for result in ann_results]
            annotated_block = [(ann[0]//self.number_of_clips, ann[1]) for ann in annotated_clips]

            counter = Counter(annotated_block)
            unique = list(counter.keys())
            counts = list(counter.values())

            ann_completed   = [u for u, c in zip(unique, counts) if c == self.number_of_clips] 
            ann_incompleted = [(u, int(c)) for u, c in zip(unique, counts) if c != self.number_of_clips]
            
            # Get video files
            files = glob.glob(self.VIDEO_BASE_PATH+'/*')
            files.sort()
            videos = []
            
            for i, file in enumerate(files):
                folder_path = file
                folder_name = file.split('/',1)[1]#file[7:]
                
                if os.path.isdir(folder_path):
                    first_frame = f"00001{self.FRAME_EXTENSION}"
                    first_frame_path = os.path.join(folder_path, first_frame)
                    
                    if os.path.exists(first_frame_path):
                        videos.append({
                            'index': i,
                            'folder': folder_name,
                            'path': folder_path
                        })
            
            num_videos   = len(videos)
            num_blocks   = num_videos//self.number_of_clips

            if len(ann_incompleted) == 0:
                possible_choices = [(block, f) for block in np.arange(num_blocks) for f in self.choices_annotatedframe]
                global_counts = [self.annotations_collection.count_documents({"globalIndex": int((p+1)*self.number_of_clips), "videoInfo.annotationFrame": f}) for p, f in possible_choices]
                possible_choices = [p for p, c in zip(possible_choices, global_counts) if c < self.annotator_per_clip]  
                possible_choices = list(set(possible_choices) - set(ann_completed))
                
                id_r = random.randint(0, len(possible_choices)-1)
                rand_number    = possible_choices[id_r][0]
                videos      = videos[int(rand_number*self.number_of_clips):int((rand_number+1)*self.number_of_clips)]
                start_index = int(rand_number*self.number_of_clips)


                for f in reversed(self.choices_annotatedframe):
                    if (rand_number, f) in set(possible_choices):
                        annotatedFrame = f
                        break

            else:


                possible_choices = [p[0][0] for p in ann_incompleted]
                possible_frames =  [p[0][1] for p in ann_incompleted]
                left_in = [p[1] for p in ann_incompleted]
                
                id_r = random.randint(0, len(possible_choices)-1)

                rand_number    = possible_choices[id_r] 
                videos         = videos[int(rand_number*self.number_of_clips)+left_in[id_r]:int((rand_number+1)*self.number_of_clips)]
                start_index    = int(rand_number*self.number_of_clips)+left_in[id_r]
                annotatedFrame = possible_frames[id_r]

            return jsonify({
                'success': True,
                'start_index': start_index,
                'total_videos': len(videos),
                'annotatedFrame': annotatedFrame,
                'videos': videos
            })

        @self.app.route('/api/video/<int:video_index>/frame/<int:frame_index>')
        def get_frame(video_index, frame_index):
            """Serve a specific frame from a video"""
            files = glob.glob(self.VIDEO_BASE_PATH+'/*')
            files.sort()
            
            folder_name = files[video_index].split('/',1)[1]#[7:]
            
            frame_name = f"{str(frame_index + 1).zfill(self.FRAME_COUNT_PADDING)}{self.FRAME_EXTENSION}"
            folder_path = os.path.join(self.VIDEO_BASE_PATH, folder_name)
            
            if not os.path.exists(os.path.join(folder_path, frame_name)):
                return jsonify({'error': 'Frame not found'}), 404
            
            return send_from_directory(folder_path, frame_name)

        @self.app.route('/api/save-annotation', methods=['POST'])
        def save_annotation():
            """Save a single annotation to MongoDB"""
            if self.db is None:
                return jsonify({'success': False, 'error': 'Database not connected'}), 500
            
            try:
                data = request.json
                data['annotator_id'] = session.get('annotator_id', 'unknown')
                data['created_at']   = datetime.utcnow()
                data['updated_at']   = datetime.utcnow()

                result = self.annotations_collection.insert_one(data)

                print(f"✅ Annotation saved with ID: {result.inserted_id}")
                
                return jsonify({
                    'success': True,
                    'message': 'Annotation saved to MongoDB',
                    'annotation_id': str(result.inserted_id)
                })
            except Exception as e:
                print(f"❌ Error saving annotation: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        #########################
        ### VALIDATION ROUTES ###
        #########################

        @self.app.route('/')
        def index():
            return render_template('validation.html')

        @self.app.route('/api/validation/detect-videos')
        def validation_detect_videos():
            """Detect available video folders"""

            try:
                ann_results = self.annotations_collection.find({"annotator_id": session['annotator_id']})
            except:
                ann_results = []

            files = glob.glob(self.VIDEO_VALIDATION_BASE_PATH+'/*')
            files.sort()
            videos = []
            
            for i, file in enumerate(files):
                folder_path = file
                folder_name = file.split('/',1)[1]#file[7:]
                
                if os.path.isdir(folder_path):
                    first_frame = f"00001{self.FRAME_EXTENSION}"
                    first_frame_path = os.path.join(folder_path, first_frame)
                    
                    if os.path.exists(first_frame_path):
                        videos.append({
                            'index': i,
                            'folder': folder_name,
                            'path': folder_path
                        })

            num_videos   = len(videos)
            id_r = random.randint(0, num_videos-1)
            videos = [videos[id_r]]
            
            return jsonify({
                'success': True,
                'start_index': 0,
                'total_videos': 1,
                'videos': videos
            })
 
 
        
        @self.app.route('/api/validation/video/<int:video_index>/frame/<int:frame_index>')
        def get_validation_frame(video_index, frame_index):
            """Serve a specific frame from a video"""
            files = glob.glob(self.VIDEO_VALIDATION_BASE_PATH+'/*')
            files.sort()
            
            folder_name = files[video_index].split('/',1)[1]#[7:]
            
            frame_name = f"{str(frame_index + 1).zfill(self.FRAME_COUNT_PADDING)}{self.FRAME_EXTENSION}"
            folder_path = os.path.join(self.VIDEO_VALIDATION_BASE_PATH, folder_name)
            
            if not os.path.exists(os.path.join(folder_path, frame_name)):
                return jsonify({'error': 'Frame not found'}), 404
            
            return send_from_directory(folder_path, frame_name)
        
        @self.app.route('/api/validation/save-annotation', methods=['POST'])
        def save_validation_annotation():
            try:
                data = request.json

                data['annotator_id'] = session.get('annotator_id', 'unknown')
                data['created_at']   = datetime.utcnow()
                data['updated_at']   = datetime.utcnow()

                gt_file = data['videoFolder']+'.txt'

                predictions = [bbox['bbox'] for bbox in data['groups']]

                with open(f'{self.GTS_VALIDATION_BASE_PATH}/{gt_file}') as f:
                    gts = f.readlines()

                gts = [gt.strip() for gt in gts]
                gts = [list(map(int, gt.split(' '))) for gt in gts]

                score = best_iou_matching(gts, predictions)
                score = (2*score)/(len(gts)+len(predictions))

                return jsonify({
                    'success': True,
                    'message': 'Annotation saved compared to GT',
                    'score': score
                })

            except Exception as e:
                print(f"❌ Error saving annotation: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500
       
        #########################
        ### YES AND NO ROUTES ###
        #########################

        @self.app.route('/api_yes_no/detect-videos')
        def yes_no_detect_videos():
            """Detect available video folders"""

            # Get completed annotations
            try:
                ann_results = self.yes_no_annotations_collection.find({"annotator_id": session['annotator_id']})
            except:
                ann_results = []

            annotated_clips = [(result['globalIndex']-1, result['videoInfo']['annotationFrame']) for result in ann_results]
            annotated_block = [(ann[0]//self.number_of_clips, ann[1]) for ann in annotated_clips]

            counter = Counter(annotated_block)
            unique = list(counter.keys())
            counts = list(counter.values())

            ann_completed   = [u for u, c in zip(unique, counts) if c == self.number_of_clips]
            ann_incompleted = [(u, int(c)) for u, c in zip(unique, counts) if c != self.number_of_clips]

            files = glob.glob(self.YES_NO_VIDEO_BASE_PATH+'/*')
            files.sort()
            videos = []

            for i, file in enumerate(files):
                folder_path = file
                folder_name = file.split('/',1)[1]#file[7:]

                if os.path.isdir(folder_path):
                    first_frame = f"00001{self.FRAME_EXTENSION}"
                    first_frame_path = os.path.join(folder_path, first_frame)

                    if os.path.exists(first_frame_path):

                        videos.append({
                            'index': i,
                            'folder': folder_name,
                            'path': folder_path,
                        })

            num_videos   = len(videos)
            num_blocks   = num_videos//self.number_of_clips

            if len(ann_incompleted) == 0:
                possible_choices = [(block, f) for block in np.arange(num_blocks) for f in self.choices_annotatedframe]
                global_counts = [self.yes_no_annotations_collection.count_documents({"globalIndex": int((p+1)*self.number_of_clips), "videoInfo.annotationFrame": f}) for p, f in possible_choices]
                possible_choices = [p for p, c in zip(possible_choices, global_counts) if c < self.annotator_per_clip]
                possible_choices = list(set(possible_choices) - set(ann_completed))

                id_r = random.randint(0, len(possible_choices)-1)
                rand_number    = possible_choices[id_r][0]
                videos      = videos[int(rand_number*self.number_of_clips):int((rand_number+1)*self.number_of_clips)]
                start_index = int(rand_number*self.number_of_clips)


                for f in self.choices_annotatedframe:
                    if (rand_number, f) in set(possible_choices):
                        annotatedFrame = f
                        break

            else:


                possible_choices = [p[0][0] for p in ann_incompleted]
                possible_frames =  [p[0][1] for p in ann_incompleted]
                left_in = [p[1] for p in ann_incompleted]

                id_r = random.randint(0, len(possible_choices)-1)

                rand_number    = possible_choices[id_r]
                videos         = videos[int(rand_number*self.number_of_clips)+left_in[id_r]:int((rand_number+1)*self.number_of_clips)]
                start_index    = int(rand_number*self.number_of_clips)+left_in[id_r]
                annotatedFrame = possible_frames[id_r]


            for ind, video in enumerate(videos):

                self.global_idx = 0
                group_file = f"groups/{annotatedFrame}_{video['folder']}.pkl"
                with open(group_file, 'rb') as f:
                    groups = pickle.load(f)

                all_boxes = []
                splits = []

                for group in groups:
                    boxes = generate_boxes(group)
                    all_boxes.extend(boxes)
                    splits.append(boxes)

                videos[ind]['splits'] = splits
                videos[ind]['boxes']  = all_boxes

            return jsonify({
                'success': True,
                'start_index': start_index,
                'total_videos': len(videos),
                'annotatedFrame': annotatedFrame,
                'videos': videos
            })


        @self.app.route('/api_yes_no/video/<int:video_index>/frame/<int:frame_index>')
        def yes_no_get_frame(video_index, frame_index):
            """Serve a specific frame from a video"""
            files = glob.glob(self.YES_NO_VIDEO_BASE_PATH+'/*')
            files.sort()

            folder_name = files[video_index].split('/',1)[1]

            frame_name = f"{str(frame_index + 1).zfill(self.FRAME_COUNT_PADDING)}{self.FRAME_EXTENSION}"
            folder_path = os.path.join(self.YES_NO_VIDEO_BASE_PATH, folder_name)

            if not os.path.exists(os.path.join(folder_path, frame_name)):
                return jsonify({'error': 'Frame not found'}), 404

            return send_from_directory(folder_path, frame_name)

        @self.app.route('/api_yes_no/save-annotation', methods=['POST'])
        def yes_no_save_annotation():
            """Save a single annotation to MongoDB"""
            if self.db is None:
                return jsonify({'success': False, 'error': 'Database not connected'}), 500

            try:
                data = request.json
                data['annotator_id'] = session.get('annotator_id', 'unknown')
                data['created_at']   = datetime.utcnow()
                data['updated_at']   = datetime.utcnow()

                result = self.yes_no_annotations_collection.insert_one(data)

                print(f"✅ Annotation saved with ID: {result.inserted_id}")

                return jsonify({
                    'success': True,
                    'message': 'Annotation saved to MongoDB',
                    'annotation_id': str(result.inserted_id)
                })
            except Exception as e:
                print(f"❌ Error saving annotation: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/yes_no_annotation')
        def yes_no_index():
            return render_template('yes_no_annotation.html')

        @self.app.route("/yes_no_thank_you")
        def yes_no_thank_you():
            return render_template("yes_no_thank_you.html")

        ##############################
        ### FINEGRAINED ANNOTATION ###
        ##############################

        @self.app.route('/finegrained_annotation')
        def finegrained_index():
            return render_template('finegrained_annotation.html')

        @self.app.route('/finegrained_thank_you')
        def finegrained_thank_you():
            return render_template('finegrained_thank_you.html')

        @self.app.route('/api_finegrained/detect-videos')
        def finegrained_detect_videos():
            """Detect available video folders"""

            # Get completed annotations
            try:
                ann_results = self.finegrained_annotations_collection.find({"annotator_id": session['annotator_id']})
            except:
                ann_results = []

            annotated_clips = [(result['globalIndex']-1, result['videoInfo']['annotationFrame']) for result in ann_results]
            annotated_block = [(ann[0]//self.number_of_clips, ann[1]) for ann in annotated_clips]

            counter = Counter(annotated_block)
            unique = list(counter.keys())
            counts = list(counter.values())

            ann_completed   = [u for u, c in zip(unique, counts) if c == self.number_of_clips] 
            ann_incompleted = [(u, int(c)) for u, c in zip(unique, counts) if c != self.number_of_clips]
            
            # Get video files
            files = glob.glob(self.VIDEO_BASE_PATH+'/*')
            files.sort()
            videos = []
            
            for i, file in enumerate(files):
                folder_path = file
                folder_name = file.split('/',1)[1]#file[7:]
                
                if os.path.isdir(folder_path):
                    first_frame = f"00001{self.FRAME_EXTENSION}"
                    first_frame_path = os.path.join(folder_path, first_frame)
                    
                    if os.path.exists(first_frame_path):
                        videos.append({
                            'index': i,
                            'folder': folder_name,
                            'path': folder_path
                        })
            
            num_videos   = len(videos)
            num_blocks   = num_videos//self.number_of_clips

            if len(ann_incompleted) == 0:
                possible_choices = [(block, f) for block in np.arange(num_blocks) for f in self.choices_annotatedframe]
                global_counts = [self.finegrained_annotations_collection.count_documents({"globalIndex": int((p+1)*self.number_of_clips), "videoInfo.annotationFrame": f}) for p, f in possible_choices]
                possible_choices = [p for p, c in zip(possible_choices, global_counts) if c < self.annotator_per_clip]  
                possible_choices = list(set(possible_choices) - set(ann_completed))
                
                id_r = random.randint(0, len(possible_choices)-1)
                rand_number    = possible_choices[id_r][0]
                videos      = videos[int(rand_number*self.number_of_clips):int((rand_number+1)*self.number_of_clips)]
                start_index = int(rand_number*self.number_of_clips)


                for f in reversed(self.choices_annotatedframe):
                    if (rand_number, f) in set(possible_choices):
                        annotatedFrame = f
                        break

            else:


                possible_choices = [p[0][0] for p in ann_incompleted]
                possible_frames =  [p[0][1] for p in ann_incompleted]
                left_in = [p[1] for p in ann_incompleted]
                
                id_r = random.randint(0, len(possible_choices)-1)

                rand_number    = possible_choices[id_r] 
                videos         = videos[int(rand_number*self.number_of_clips)+left_in[id_r]:int((rand_number+1)*self.number_of_clips)]
                start_index    = int(rand_number*self.number_of_clips)+left_in[id_r]
                annotatedFrame = possible_frames[id_r]


            print(start_index)
            print(annotatedFrame)

            return jsonify({
                'success': True,
                'start_index': start_index,
                'total_videos': len(videos),
                'annotatedFrame': annotatedFrame,
                'videos': videos
            })



        @self.app.route('/api_finegrained/video/<int:video_index>/frame/<int:frame_index>')
        def finegrained_get_frame(video_index, frame_index):
            """Serve a frame from a video"""
            # Get video info from annotations

#            video_info = None
#            for ann in self.COARSE_ANNOTATIONS['annotations']:
#                
#                if ann['videoIndex'] == video_index:
#                    video_info = ann
#                    break
#        
#            if not video_info:
#                return jsonify({'error': 'Video not found'}), 404
#        
#            actual_video_index = video_info['videoIndex']
#            folder_name = f"clip_{actual_video_index:04d}"
#            frame_name = f"{int(frame_index) + 1:05d}.jpeg"
#            folder_path = self.VIDEOS_DIR / folder_name
#            frame_path = folder_path / frame_name
#        
#            if not frame_path.exists():
#                return jsonify({'error': 'Frame not found'}), 404
#
            files = glob.glob(self.VIDEO_BASE_PATH+'/*')
            files.sort()

            folder_name = files[video_index-1].split('/',1)[1]#[7:]
            print(folder_name)

            frame_name = f"{str(frame_index + 1).zfill(self.FRAME_COUNT_PADDING)}{self.FRAME_EXTENSION}"
            print(frame_name)
            folder_path = os.path.join(self.VIDEO_BASE_PATH, folder_name)

            if not os.path.exists(os.path.join(folder_path, frame_name)):
                return jsonify({'error': 'Frame not found'}), 404
       
            print("REPORTS")
            print(frame_name)
            print(folder_path)
            print("REPORTS")
            
            return send_from_directory(str(folder_path), frame_name)
        
        @self.app.route('/api_finegrained/detect-frame/<int:video_index>/frame/<int:frame_index>')
        def finegrained_detect_frame(video_index, frame_index):
            """
            Run fine-grained detection on annotation frame
        
            Algorithm:
            1. Load coarse groups from annotation
            2. Sort by area (smallest first)
            3. Run YOLO person detection on first frame
            4. Assign detected persons to groups (center-point method)
            5. Return detections with group labels
            """
        
            files = glob.glob(self.VIDEO_BASE_PATH+'/*')
            files.sort()
            folder_name = files[video_index-1].split('/',1)[1]#[7:]
            video_index = int(folder_name.split("_")[-1])

            print(video_index)
            # Find video annotation
            video_info = None
            for ann in self.COARSE_ANNOTATIONS['annotations']:
                if ann['videoIndex'] == video_index and ann['videoInfo']['annotationFrame'] == frame_index:
                    video_info = ann
                    break
        
            if not video_info:
                return jsonify({'error': 'Video not found'}), 404
        
            actual_video_index = video_info['videoIndex']
            coarse_groups = video_info['groups']
            annotation_frame = video_info['videoInfo']['annotationFrame']
            print(annotation_frame)
        
            # Load annotation frame
            print(str(annotation_frame).zfill(5)+".jpeg")
            frame_path = self.VIDEOS_DIR / f"clip_{actual_video_index:04d}" / f"{annotation_frame+1:05d}.jpeg"

            if not frame_path.exists():
                return jsonify({'error': 'Frame not found'}), 404
        
            # Load and process image
            image = cv2.imread(str(frame_path))
            if image is None:
                return jsonify({'error': 'Failed to load image'}), 500
        
            height, width = image.shape[:2]
        
            # Load detections from cache (preprocessed YOLO results)
            cache_key = str(video_index)+'_'+str(annotation_frame)
            print(cache_key)
            if cache_key in self.DETECTIONS_CACHE:
                print(f"[DETECT] Loading cached detections for video {actual_video_index}")
                detections = []
                for det in self.DETECTIONS_CACHE[cache_key]['detections']:
                    detections.append({
                        'bbox': det['bbox'],
                        'assigned': False
                    })
            else:
                print(f"[ERROR] No cached detections for video {actual_video_index}!")
                print(f"[INFO] Please run: python preprocess_detections.py")
                return jsonify({'error': 'Detections cache not found. Please run preprocess_detections.py first'}), 500
        
            print(f"[DETECT] Found {len(detections)} persons")
        
            # Sort coarse groups by area (smallest first)
            def bbox_area(bbox):
                x1, y1, x2, y2 = bbox
                return max(0, (x2 - x1) * (y2 - y1))
        
            coarse_groups_sorted = sorted(coarse_groups, key=lambda g: bbox_area(g['bbox']))
        
            # Generate boxes data for each detection
            boxes = []
        
            # First, assign detections to groups
            for group in coarse_groups_sorted:
                group_id = group['groupId']
                gx1, gy1, gx2, gy2 = group['bbox']
        
                # Find detections inside this group
                for detection in detections:
                    if detection['assigned']:
                        continue
        
                    dx1, dy1, dx2, dy2 = detection['bbox']
                    center_x = (dx1 + dx2) / 2
                    center_y = (dy1 + dy2) / 2
        
                    # Check if center is inside group bbox
                    if gx1 <= center_x <= gx2 and gy1 <= center_y <= gy2:
                        boxes.append({
                            'id': f'box_{len(boxes):05d}',
                            'tl_x': int(dx1),
                            'tl_y': int(dy1),
                            'br_x': int(dx2),
                            'br_y': int(dy2),
                            'confidence': 5,
                            'label': f'group_{group_id}'
                        })
                        detection['assigned'] = True
        
            # Add unassigned detections as individuals
            for detection in detections:
                if not detection['assigned']:
                    dx1, dy1, dx2, dy2 = detection['bbox']
                    boxes.append({
                        'id': f'box_{len(boxes):05d}',
                        'tl_x': int(dx1),
                        'tl_y': int(dy1),
                        'br_x': int(dx2),
                        'br_y': int(dy2),
                        'confidence': 3,
                        'label': 'individual'
                    })
        
            print(f"[RESULT] Generated {len(boxes)} boxes")
        
            return jsonify({
                'success': True,
                'videoIndex': video_index,
                'folder': video_info['videoFolder'],
                'frameWidth': width,
                'frameHeight': height,
                'boxes': boxes,
                'coarseGroups': coarse_groups,
                'totalFrames': video_info['videoInfo']['totalFrames']
            })

        @self.app.route('/api_finegrained/coarse-groups/<int:video_index>/frame/<int:frame_index>')
        def get_coarse_groups(video_index, frame_index):
            """Get coarse group annotations for a video"""
            try:

                files = glob.glob(self.VIDEO_BASE_PATH+'/*')
                files.sort()
                folder_name = files[video_index-1].split('/',1)[1]#[7:]
                video_index = int(folder_name.split("_")[-1])

                #print(self.COARSE_ANNOTATIONS['annotations'])
                # Find video annotation
                video_info = None
                for ann in self.COARSE_ANNOTATIONS['annotations']:
                    if ann['videoIndex'] == video_index and ann['videoInfo']['annotationFrame'] == frame_index:
                        video_info = ann
                        break

                if not video_info:
                    return jsonify({'error': 'Video not found'}), 404
        
                coarse_groups = video_info.get('groups', [])
                return jsonify({'groups': coarse_groups})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api_finegrained/save-annotation', methods=['POST'])
        def finegrained_save_annotation():
            """Save a single annotation to MongoDB"""
            if self.db is None:
                return jsonify({'success': False, 'error': 'Database not connected'}), 500

            try:
                data = request.json
                data['annotator_id'] = session.get('annotator_id', 'unknown')
                data['created_at']   = datetime.utcnow()
                data['updated_at']   = datetime.utcnow()

                result = self.finegrained_annotations_collection.insert_one(data)

                print(f"✅ Annotation saved with ID: {result.inserted_id}")

                return jsonify({
                    'success': True,
                    'message': 'Annotation saved to MongoDB',
                    'annotation_id': str(result.inserted_id)
                })
            except Exception as e:
                print(f"❌ Error saving annotation: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        ####################
        ### MISCELANIOUS ###       
        ####################

        @self.app.route('/my-annotator-id')
        def show_annotator_id():
            """Display the annotator ID in a user-friendly page"""
            if 'annotator_id' not in session:
                session['annotator_id'] = str(uuid.uuid4())
                session.permanent = True
            return render_template('annotator_id.html', annotator_id=session['annotator_id'])
         
        @self.app.route('/api/save-all-annotations', methods=['POST'])
        def save_all_annotations():
            """Save all annotations in a single file"""
            if self.db is None:
                return jsonify({'success': False, 'error': 'Database not connected'}), 500
            
            datas = request.json
            try:
                for data in datas['annotations']:
                    # Add server-side metadata
                    data['annotator_id'] = session.get('annotator_id', 'unknown')
                    data['created_at']   = datetime.utcnow()
                    data['updated_at']   = datetime.utcnow()
                    
                    # Insert into MongoDB
                    result = self.annotations_collection.insert_one(data)
                    print(f"✅ Annotation saved with ID: {result.inserted_id}")
                
                return jsonify({
                    'success': True,
                    'message': 'All annotations save to DB',
                    'annotation_id': str(result.inserted_id)
                })
            except Exception as e:
                print(f"❌ Error saving annotation: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

video_app = VideoAnnotationApp()
app = video_app.app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
        
