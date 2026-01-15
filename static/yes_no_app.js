class VideoAnnotationTool {
    constructor() {
        this.currentVideoIndex = 0;
        this.globalVideoIndex = 0;
        this.totalVideos = 0;
        this.videos = [];
        this.totalFrames = 50;
        this.frameRate = 5;

        this.NORMALIZED_WIDTH = 1920;
        this.NORMALIZED_HEIGHT = 1080;
        
        this.allAnnotations = [];
        this.currentBoxes = [];
        this.currentSplits = [];

        // Pairwise annotation state
        this.boxPairs = [];
        this.currentPairIndex = 0;
        this.groupAnnotations = [];
        
        this.isPlaying = false;
        this.currentFrame = 0;
        this.hasWatchedVideo = false;
        this.videoPlayStartTime = null;
        this.totalWatchTime = 0;
        this.playbackSpeed = 1;
        this.idx1 = 0
        this.idx2 = 0
        
        this.annotatorId = null;

        this.videoActionHistory = [];
        this.sessionStartTime = new Date().toISOString();
        this.videoLoadTime = null;

        this.resizeTimeout = null;

        this.loadAnnotatorId();
        this.initElements();
        this.bindEvents();
        this.loadVideos();
    }

    logVideoAction(action, metadata = {}) {
        const timestamp = new Date().toISOString();
        const timeSinceVideoLoad = this.videoLoadTime
            ? Date.now() - new Date(this.videoLoadTime).getTime()
            : 0;

        const actionLog = {
            timestamp,
            timeSinceVideoLoad,
            action,
            currentFrame: this.currentFrame,
            isPlaying: this.isPlaying,
            playbackSpeed: this.playbackSpeed,
            ...metadata
        };

        this.videoActionHistory.push(actionLog);
        console.log('Video Action:', actionLog);
    }

    
    async loadAnnotatorId() {
        try {
            const response = await fetch('/api/get-annotator-id');
            const data = await response.json();
            this.annotatorId = data.annotator_id;
            console.log('Annotator ID:', this.annotatorId);
        } catch (error) {
            console.error('Error getting annotator ID:', error);
        }
    }

    async loadVideos() {
        try {
            const response = await fetch('/api_yes_no/detect-videos');
            const data = await response.json();
            
            if (data.success) {
                this.globalVideoIndex = data.start_index;
                this.videos = data.videos;
                this.totalVideos = data.total_videos;
                this.totalVideoNum.textContent = this.totalVideos;
                this.annotatedFrame = data.annotatedFrame;
                
                if (this.totalVideos > 0) {
                    this.loadVideo(0);
                } else {
                    alert('No video folders found! Please add video folders to the videos directory.');
                }
            }
        } catch (error) {
            console.error('Error loading videos:', error);
            alert('Error loading videos. Please check the console.');
        }
    }
    
    initElements() {
        this.annotationCanvas = document.getElementById('annotationCanvas');
        this.annotationImage = document.getElementById('annotationImage');
        this.videoCanvas = document.getElementById('videoCanvas');
        this.videoImage = document.getElementById('videoImage');
        this.playBtn = document.getElementById('playBtn');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.frameCounter = document.getElementById('frameCounter');
        this.totalFramesEl = document.getElementById('totalFrames');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.selectionCoords = document.getElementById('selectionCoords');
        this.submitBtn = document.getElementById('submitBtn');
        this.playbackStatus = document.getElementById('playbackStatus');
        this.watchedStatus = document.getElementById('watchedStatus');
        this.currentFrameNum = document.getElementById('currentFrameNum');
        this.currentVideoNum = document.getElementById('currentVideoNum');
        this.totalVideoNum = document.getElementById('totalVideoNum');
        this.boxCount = document.getElementById('boxCount');
        this.videoScrubber = document.getElementById('videoScrubber');
        this.speedButtons = document.querySelectorAll('.speed-btn');
        
        // Pairwise annotation elements
        this.yesBtn = document.getElementById('yesBtn');
        this.noBtn = document.getElementById('noBtn');
        this.pairProgress = document.getElementById('pairProgress');
        this.pairQuestion = document.getElementById('pairQuestion');
    }
    
    bindEvents() {
        this.playBtn.addEventListener('click', () => this.play());
        this.pauseBtn.addEventListener('click', () => this.pause());
        
        this.submitBtn.addEventListener('click', () => this.submitAnnotation());
        
        // Pairwise annotation buttons
        this.yesBtn.addEventListener('click', () => this.answerPair(true));
        this.noBtn.addEventListener('click', () => this.answerPair(false));
        
        this.videoScrubber.addEventListener('input', (e) => {
            const previousFrame = this.currentFrame;
            const newFrame = parseInt(e.target.value);

            this.logVideoAction('scrubber_seek', {
                fromFrame: previousFrame,
                toFrame: newFrame,
                frameDelta: newFrame - previousFrame
            });

            this.hasWatchedVideo = true;
            this.updateWatchedStatus();
            this.currentFrame = newFrame;
            this.loadVideoFrame(this.getCurrentVideo().index, this.currentFrame);
            this.updateDisplay();
        });

        
        // Speed controls
        this.speedButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const speed = parseFloat(e.target.dataset.speed);
                this.setPlaybackSpeed(speed);
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                return;
            }
            
            switch(e.key) {
                case 'Shift':
                    e.preventDefault();
                    this.isPlaying ? this.pause() : this.play();
                    break;
                case 'ArrowRight':
                    this.hasWatchedVideo = true;
                    this.updateWatchedStatus();
                    e.preventDefault();
                    if (e.shiftKey) {
                        this.skipFrames(5);
                    } else {
                        this.nextFrame();
                    }
                    break;
                case 'ArrowLeft':
                    this.hasWatchedVideo = true;
                    this.updateWatchedStatus();
                    e.preventDefault();
                    if (e.shiftKey) {
                        this.skipFrames(-5);
                    } else {
                        this.previousFrame();
                    }
                    break;
                case 'y':
                case 'Y':
                    e.preventDefault();
                    this.answerPair(true);
                    break;
                case 'n':
                case 'N':
                    e.preventDefault();
                    this.answerPair(false);
                    break;
            }
        });
        
        // Wait for annotation image to load before drawing boxes
        this.annotationImage.addEventListener('load', () => {
            this.drawBoundingBoxes(this.boxPairs[this.currentPairIndex]);
        });

        // Window resize handler
        window.addEventListener('resize', () => {
            clearTimeout(this.resizeTimeout);
            this.resizeTimeout = setTimeout(() => {
                console.log(this.idx1);
                console.log(this.idx2);
                this.drawBoundingBoxes(this.boxPairs[this.currentPairIndex]);
            }, 100);
        });
    }
    
    generateBoxPairs() {
        this.boxPairs = [];

        const l = this.currentSplits.length ?? 0;
        let real = 0;
        for (let l_i = 0; l_i < l; l_i++)
        {
            const n = this.currentSplits[l_i].length;
            console.log(n);
            let arr = [];
            for (let i = 0; i < n; i++) 
                arr.push(real+i)
            this.boxPairs.push(arr)
            real= real + n;
        }
        this.currentPairIndex = 0;
        this.groupAnnotations = [];
        
        //console.log(`Generated ${this.boxPairs.length} pairs from ${n} boxes`);
    }
    
    answerPair(isGroup) {
        if (this.currentPairIndex >= this.boxPairs.length) {
            return;
        }
        
        this.groupAnnotations.push({
            group: this.boxPairs[this.currentPairIndex],
            is_group: isGroup,
            timestamp: new Date().toISOString()
        });
        
        this.currentPairIndex++;
        
        if (this.currentPairIndex < this.boxPairs.length) {
            this.showCurrentPair();
        } else {
            this.finishPairwiseAnnotation();
        }
    }
    
    showCurrentPair() {
        if (this.currentPairIndex >= this.boxPairs.length) {
            return;
        }

        console.log(this.boxPairs)
        
        const [idx1, idx2] = this.boxPairs[this.currentPairIndex];
        
        this.idx1 = idx1 
        this.idx2 = idx2

        // Update progress
        this.pairProgress.textContent = `Group ${this.currentPairIndex + 1} of ${this.boxPairs.length}`;
        this.pairQuestion.textContent = `Are all the annotated persons in the same group?`;
        
        // Redraw only the current pair
        this.drawBoundingBoxes(this.boxPairs[this.currentPairIndex]);
    }
    
    finishPairwiseAnnotation() {
        this.pairProgress.textContent = `All ${this.boxPairs.length} pairs completed! ✅`;
        this.pairQuestion.textContent = 'You can now submit your annotation.';
        
        // Show all boxes again
        this.drawBoundingBoxes();
        
        // Enable submit button
        this.submitBtn.disabled = false;
        
        // Hide yes/no buttons
        this.yesBtn.style.display = 'none';
        this.noBtn.style.display = 'none';
    }
    
    loadVideo(index) {
        if (index >= this.totalVideos) {
            return;
        }
        
        this.currentVideoIndex = index;
        this.globalVideoIndex += 1;
        this.currentVideoNum.textContent = index + 1;
        this.currentFrameNum.textContent = this.annotatedFrame;
        
        const video = this.videos[index];

        this.videoActionHistory = [];
        this.videoLoadTime = new Date().toISOString();
        this.logVideoAction('video_loaded', {
            videoIndex: this.currentVideoIndex,
            globalVideoIndex: this.globalVideoIndex,
            videoFolder: video.folder,
            annotatedFrame: this.annotatedFrame
        });

        this.currentBoxes  = video.boxes || [];
        this.currentSplits = video.splits || [];
        this.boxCount.textContent = this.currentBoxes.length;
        
        this.annotationImage.src = `/api_yes_no/video/${video.index}/frame/${this.annotatedFrame}`;
        this.loadVideoFrame(video.index, 0);
        
        this.resetVideoState();
        this.updateDisplay();
        this.updateSelectionInfo();

        const max_size = this.currentSplits.reduce(
            (max, sublista) => Math.max(max, sublista.length),
            0
        );

        console.log(this.currentSplits)
        console.log(max_size)
        
        // Initialize pairwise annotation if there are boxes
        if (max_size >= 2) {
            this.generateBoxPairs();
            this.showCurrentPair();
            this.yesBtn.style.display = 'inline-flex';
            this.noBtn.style.display = 'inline-flex';
            this.submitBtn.disabled = true;
        } else {
            // No pairs to annotate
            this.pairProgress.textContent = this.currentBoxes.length === 0 ? 'No boxes to annotate' : 'Only 1 box - no pairs to compare';
            this.pairQuestion.textContent = 'You can submit directly.';
            this.yesBtn.style.display = 'none';
            this.noBtn.style.display = 'none';
            this.submitBtn.disabled = false;
        }
    }
    
    drawBoundingBoxes(visibleIndices = null) {
        // Clear previous boxes
        const existingBoxes = this.annotationCanvas.querySelectorAll('.selection-box');
        existingBoxes.forEach(box => box.remove());
        
        if (this.currentBoxes.length === 0) {
            return;
        }
        
        // Get image dimensions for scaling
        const rect = this.annotationCanvas.getBoundingClientRect();
        const imgRect = this.annotationImage.getBoundingClientRect();
        
        // Draw each box (or only visible ones)
        this.currentBoxes.forEach((box, index) => {
            // If visibleIndices is provided, only show those boxes
            if (visibleIndices !== null && !visibleIndices.includes(index)) {
                return;
            }
            
            const boxElement = document.createElement('div');
            boxElement.className = 'selection-box';
            boxElement.id = box.id;

            const x1 = Math.min(box.tl_x1, box.br_x1);
            const y1 = Math.min(box.tl_y1, box.br_y1);
            const x2 = Math.max(box.tl_x1, box.br_x1);
            const y2 = Math.max(box.tl_y1, box.br_y1);

            const displayX1 = (x1 / this.NORMALIZED_WIDTH) * imgRect.width;
            const displayY1 = (y1 / this.NORMALIZED_HEIGHT) * imgRect.height;
            const displayX2 = (x2 / this.NORMALIZED_WIDTH) * imgRect.width;
            const displayY2 = (y2 / this.NORMALIZED_HEIGHT) * imgRect.height;

            const canvasX = imgRect.left - rect.left + displayX1;
            const canvasY = imgRect.top - rect.top + displayY1;
            const width = displayX2 - displayX1;
            const height = displayY2 - displayY1;

            boxElement.style.left = `${canvasX}px`;
            boxElement.style.top = `${canvasY}px`;
            boxElement.style.width = `${width}px`;
            boxElement.style.height = `${height}px`;

            const label = document.createElement('div');
            label.className = 'box-label';
            label.innerHTML = `
                <span>Person ${index + 1}</span>
            `;
            boxElement.appendChild(label);
            this.annotationCanvas.appendChild(boxElement);
        });
    }
    
    updateSelectionInfo() {
        if (this.currentBoxes.length === 0) {
            this.selectionCoords.innerHTML = 'No bounding boxes for this video';
            return;
        }
        
        const html = this.currentBoxes.map((box, index) => `
            <div class="box-item">
                <strong>Person ${index + 1}</strong><br>
                Coordinates: (${box.tl_x1}, ${box.tl_y1}, ${box.br_x1}, ${box.br_y1})<br>
            </div>
        `).join('');
        
        this.selectionCoords.innerHTML = html;
    }
    
    loadVideoFrame(videoIndex, frameIndex) {
        this.videoImage.src = `/api_yes_no/video/${videoIndex}/frame/${frameIndex}`;
    }
    
    getCurrentVideo() {
        return this.videos[this.currentVideoIndex];
    }
    
    setPlaybackSpeed(speed) {
        const previousSpeed = this.playbackSpeed;
        this.playbackSpeed = speed;

        this.logVideoAction('playback_speed_changed', {
            fromSpeed: previousSpeed,
            toSpeed: speed
        });
        
        this.speedButtons.forEach(btn => {
            if (parseFloat(btn.dataset.speed) === speed) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        if (this.isPlaying) {
            this.pause();
            this.play();
        }
    }
    
    nextFrame() {
        if (this.isPlaying) {
            this.pause();
        }

        const previousFrame = this.currentFrame;
        this.currentFrame = Math.min(this.currentFrame + 1, this.totalFrames - 1);

        this.logVideoAction('frame_next', {
            fromFrame: previousFrame,
            toFrame: this.currentFrame,
            method: 'keyboard_or_button'
        });

        
        this.loadVideoFrame(this.getCurrentVideo().index, this.currentFrame);
        this.updateDisplay();
        this.videoScrubber.value = this.currentFrame;
    }
    
    previousFrame() {
        if (this.isPlaying) {
            this.pause();
        }

        const previousFrame = this.currentFrame;
        this.currentFrame = Math.max(this.currentFrame - 1, 0);

        this.logVideoAction('frame_previous', {
            fromFrame: previousFrame,
            toFrame: this.currentFrame,
            method: 'keyboard_or_button'
        });

        this.loadVideoFrame(this.getCurrentVideo().index, this.currentFrame);
        this.updateDisplay();
        this.videoScrubber.value = this.currentFrame;
    }
    
    skipFrames(count) {
        if (this.isPlaying) {
            this.pause();
        }

        const previousFrame = this.currentFrame;
        this.currentFrame = Math.max(0, Math.min(this.currentFrame + count, this.totalFrames - 1));

        this.logVideoAction('frame_skip', {
            fromFrame: previousFrame,
            toFrame: this.currentFrame,
            skipCount: count,
            method: 'keyboard'
        });

        this.loadVideoFrame(this.getCurrentVideo().index, this.currentFrame);
        this.updateDisplay();
        this.videoScrubber.value = this.currentFrame;
    }
    
    play() {
        if (!this.hasWatchedVideo) {
            this.hasWatchedVideo = true;
            this.updateWatchedStatus();
       }
        
        if (!this.videoPlayStartTime) {
            this.videoPlayStartTime = Date.now();
        }

        this.logVideoAction('play', {
            startFrame: this.currentFrame,
            playbackSpeed: this.playbackSpeed
        });
        
        this.isPlaying = true;
        this.playBtn.style.display = 'none';
        this.pauseBtn.style.display = 'inline-flex';
        this.statusIndicator.className = 'status-indicator status-playing';
        
        const video = this.getCurrentVideo();
        this.playInterval = setInterval(() => {
            this.currentFrame++;
            if (this.currentFrame >= this.totalFrames) {
                this.currentFrame = 0;
            }
            this.loadVideoFrame(video.index, this.currentFrame);
            this.updateDisplay();
            this.videoScrubber.value = this.currentFrame;
        }, 1000 / (this.frameRate * this.playbackSpeed));
    }
    
    pause() {
        if (this.videoPlayStartTime) {
            this.totalWatchTime += (Date.now() - this.videoPlayStartTime);
            this.videoPlayStartTime = null;
        }

        this.logVideoAction('pause', {
            pausedAtFrame: this.currentFrame,
            totalWatchTimeMs: this.totalWatchTime
        });
        
        this.isPlaying = false;
        this.playBtn.style.display = 'inline-flex';
        this.pauseBtn.style.display = 'none';
        this.statusIndicator.className = 'status-indicator status-paused';
        
        if (this.playInterval) {
            clearInterval(this.playInterval);
        }
    }
    
    updateWatchedStatus() {
        if (this.hasWatchedVideo) {
            this.watchedStatus.innerHTML = '✅ Video Watched';
            this.playbackStatus.className = 'playback-status status-watched';
        } else {
            this.watchedStatus.innerHTML = '⏸️ Video Not Watched';
            this.playbackStatus.className = 'playback-status status-not-watched';
        }
    }
    
    updateDisplay() {
        this.frameCounter.textContent = this.currentFrame + 1;
        this.totalFramesEl.textContent = this.totalFrames;
        this.videoScrubber.max = this.totalFrames - 1;
        this.videoScrubber.value = this.currentFrame;
    }
    
    async submitAnnotation() {
        if (this.isPlaying) {
            this.pause();
        }
        
        const annotation = {
            videoIndex: this.currentVideoIndex,
            globalIndex: this.globalVideoIndex,
            videoFolder: this.getCurrentVideo().folder,
            boxes: this.currentBoxes,
            groupAnnotations: this.groupAnnotations,
            totalPairs: this.boxPairs.length,
            watchTime: this.totalWatchTime,
            timestamp: new Date().toISOString(),
            videoInfo: {
                totalFrames: this.totalFrames,
                annotationFrame: this.annotatedFrame,
                coordinateSystem: 'normalized',
                normalizedWidth: this.NORMALIZED_WIDTH,
                normalizedHeight: this.NORMALIZED_HEIGHT
            },
            videoActionHistory: this.videoActionHistory,
            sessionStartTime: this.sessionStartTime,
            videoLoadTime: this.videoLoadTime,
            annotationDuration: Date.now() - new Date(this.videoLoadTime).getTime()
        };

        try {
            await this.saveAnnotation(annotation);
            if (this.currentVideoIndex < this.totalVideos - 1) {
                this.loadVideo(this.currentVideoIndex + 1);
            } else {
                window.location.href = "/yes_no_thank_you";
            }
        } catch (error) {
            console.error('Error saving annotation:', error);
            alert('Error saving annotation. Please check the console.');
        }

        
    }

    async saveAnnotation(annotation){
        try{
            await fetch('/api_yes_no/save-annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(annotation)
            });
        } catch (error){
            console.error('Error saving annotation:', error);
        }
    }

    resetVideoState() {
        this.hasWatchedVideo = false;
        this.totalWatchTime = 0;
        this.videoPlayStartTime = null;
        this.currentFrame = 0;
        
        if (this.isPlaying) {
            this.pause();
        }
        
        this.updateWatchedStatus();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VideoAnnotationTool();
});
