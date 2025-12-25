from flask import Flask, render_template, jsonify, request, send_from_directory, session
from scipy.optimize import linear_sum_assignment

import uuid
import os
import json
import glob
import random
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

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
        self.annotator_per_clip = 5
        self.number_of_clips = 15
        self.VIDEO_BASE_PATH = 'videos'
        self.VIDEO_VALIDATION_BASE_PATH = 'validation_videos'
        self.GTS_VALIDATION_BASE_PATH = 'validation_gts'
        self.FRAME_EXTENSION = '.jpeg'
        self.FRAME_PADDING = 4
        self.FRAME_COUNT_PADDING = 5

        # MongoDB setup
        try:
            self.client = MongoClient(self.MONGO_URI)
            self.db = self.client[self.MONGO_DB]
            self.annotations_collection = self.db['annotations']
            print(f"✅ Connected to MongoDB: {self.MONGO_DB}")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None

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
            print('entro')
            if 'annotator_id' not in session:
                session['annotator_id'] = str(uuid.uuid4())
                session.permanent = True
            return jsonify({'annotator_id': session['annotator_id']})
         
        @self.app.route('/')
        def index():
            return render_template('validation.html')
        
        @self.app.route('/annotation')
        def annotation():
            return render_template('annotation.html')
        
        @self.app.route("/thank_you")
        def thank_you():
            return render_template("thank_you.html")

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
 
        
        @self.app.route('/api/detect-videos')
        def detect_videos():
            """Detect available video folders"""

            # Get completed annotations
            try:
                ann_results = self.annotations_collection.find({"annotator_id": session['annotator_id']})
            except:
                ann_results = []

            annotated_clips = [result['globalIndex']-1 for result in ann_results]
            
            annotated_block = [ann//self.number_of_clips for ann in annotated_clips]
            unique, counts = np.unique(annotated_block, return_counts=True)
            ann_completed   = [int(u) for u, c in zip(unique, counts) if c == self.number_of_clips] 
            ann_incompleted = [(int(u), int(c)) for u, c in zip(unique, counts) if c != self.number_of_clips]
            
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
                possible_choices = np.arange(num_blocks)
                global_counts = [self.annotations_collection.count_documents({"globalIndex": int((p+1)*self.number_of_clips)}) 
                                for p in possible_choices]
                possible_choices = [p for p, c in zip(possible_choices, global_counts) if c < self.annotator_per_clip]
                possible_choices = np.setdiff1d(possible_choices, ann_completed)
                
                id_r = random.randint(0, len(possible_choices)-1)
                rand_number = possible_choices[id_r] 
                videos      = videos[int(rand_number*self.number_of_clips):int((rand_number+1)*self.number_of_clips)]
                start_index = int(rand_number*self.number_of_clips)
            else:
                possible_choices = [p[0] for p in ann_incompleted]
                left_in = [p[1] for p in ann_incompleted]
                
                id_r = random.randint(0, len(possible_choices)-1)
                rand_number = possible_choices[id_r] 
                videos      = videos[int(rand_number*self.number_of_clips)+left_in[id_r]:int((rand_number+1)*self.number_of_clips)]
                start_index = int(rand_number*self.number_of_clips)
            
            return jsonify({
                'success': True,
                'start_index': start_index,
                'total_videos': len(videos),
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
        
