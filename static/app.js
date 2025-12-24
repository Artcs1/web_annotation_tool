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
        
        this.isPlaying = false;
        this.currentFrame = 0;
        this.hasWatchedVideo = false;
        this.videoPlayStartTime = null;
        this.totalWatchTime = 0;
        this.playbackSpeed = 1;
        
        this.selectionBoxes = [];
        this.isDrawing = false;
        this.isResizing = false;
        this.currentBox = null;
        this.resizeHandle = null;
        
        this.annotatorId = null;
        this.loadAnnotatorId();

        this.initElements();
        this.bindEvents();
        this.loadVideos();
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
            const response = await fetch('/api/detect-videos');
            const data = await response.json();
            
            if (data.success) {

                this.globalVideoIndex = data.start_index;
                this.videos = data.videos;
                this.totalVideos = data.total_videos;
                this.totalVideoNum.textContent = this.totalVideos;
                
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
        this.currentVideoNum = document.getElementById('currentVideoNum');
        this.totalVideoNum = document.getElementById('totalVideoNum');
        this.boxCount = document.getElementById('boxCount');
        this.videoScrubber = document.getElementById('videoScrubber');
        this.speedButtons = document.querySelectorAll('.speed-btn');
    }
    
    bindEvents() {
        this.playBtn.addEventListener('click', () => this.play());
        this.pauseBtn.addEventListener('click', () => this.pause());
        
        this.annotationCanvas.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this.handleMouseDown(e);
        });
        this.annotationCanvas.addEventListener('mousemove', (e) => {
            e.preventDefault();
            this.handleMouseMove(e);
        });
        this.annotationCanvas.addEventListener('mouseup', (e) => {
            e.preventDefault();
            this.handleMouseUp(e);
        });
        this.annotationCanvas.addEventListener('mouseleave', (e) => {
            if (this.isDrawing || this.isResizing) {
                this.handleMouseUp(e);
            }
        });
        
        this.submitBtn.addEventListener('click', () => this.submitAnnotation());
        
        // Scrubber controls
        this.videoScrubber.addEventListener('input', (e) => {
            this.hasWatchedVideo = true;
            this.updateWatchedStatus();
            this.currentFrame = parseInt(e.target.value);
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
            }
        });
        
        window.addEventListener('resize', () => this.repositionAllBoxes());
        this.annotationImage.addEventListener('load', () => this.repositionAllBoxes());
    }
    
    loadVideo(index) {
        if (index >= this.totalVideos) {
            return;
        }
        
        this.currentVideoIndex = index;
        this.globalVideoIndex +=1;
        this.currentVideoNum.textContent = index + 1;
        
        const video = this.videos[index];
        
        this.annotationImage.src = `/api/video/${video.index}/frame/0`;
        this.loadVideoFrame(video.index, 0);
        
        this.clearAllBoxes();
        this.resetVideoState();
        this.updateDisplay();
    }
    
    loadVideoFrame(videoIndex, frameIndex) {
        this.videoImage.src = `/api/video/${videoIndex}/frame/${frameIndex}`;
    }
    
    getCurrentVideo() {
        return this.videos[this.currentVideoIndex];
    }
    
    handleMouseDown(e) {
        if (e.target.classList.contains('box-handle')) {
            this.startResize(e);
            return;
        }
        
        if (e.target.classList.contains('box-delete') || 
            e.target.classList.contains('box-confidence-btn') ||
            e.target.closest('.box-confidence-container')) {
            return;
        }
        
        // Always start drawing when clicking on the canvas
        this.startDrawing(e);
    }
    
    handleMouseMove(e) {
        if (this.isDrawing) {
            this.updateDrawing(e);
        } else if (this.isResizing) {
            this.updateResize(e);
        }
    }
    
    handleMouseUp(e) {
        if (this.isDrawing) {
            this.endDrawing(e);
        } else if (this.isResizing) {
            this.endResize();
        }
    }
    
    startDrawing(e) {
        const imgRect = this.annotationImage.getBoundingClientRect();
        
        if (!imgRect.width || !imgRect.height) {
            return;
        }
        
        const pixelX = e.clientX - imgRect.left;
        const pixelY = e.clientY - imgRect.top;
        
        const normalizedX = (pixelX / imgRect.width) * this.NORMALIZED_WIDTH;
        const normalizedY = (pixelY / imgRect.height) * this.NORMALIZED_HEIGHT;
        
        this.isDrawing = true;
        this.annotationCanvas.style.cursor = 'crosshair';
        
        this.currentBox = {
            id: Date.now(),
            startX: normalizedX,
            startY: normalizedY,
            endX: normalizedX,
            endY: normalizedY,
            element: this.createBoxElement()
        };
        
        this.annotationCanvas.appendChild(this.currentBox.element);
        this.updateBoxPosition(this.currentBox);
    }
    
    updateDrawing(e) {
        if (!this.isDrawing || !this.currentBox) return;
        
        const imgRect = this.annotationImage.getBoundingClientRect();
        const pixelX = e.clientX - imgRect.left;
        const pixelY = e.clientY - imgRect.top;

        console.log(pixelX)
        console.log(pixelY)
        
        this.currentBox.endX = (pixelX / imgRect.width) * this.NORMALIZED_WIDTH;
        this.currentBox.endY = (pixelY / imgRect.height) * this.NORMALIZED_HEIGHT;

            
        this.updateBoxPosition(this.currentBox);
    }
    
    endDrawing(e) {
        if (!this.isDrawing || !this.currentBox) return;
        
        this.isDrawing = false;
        this.annotationCanvas.style.cursor = 'default';
        
        const width = Math.abs(this.currentBox.endX - this.currentBox.startX);
        const height = Math.abs(this.currentBox.endY - this.currentBox.startY);

        this.currentBox = {
            ...this.currentBox,
            startX: Math.min(this.currentBox.startX, this.currentBox.endX),
            endX:   Math.max(this.currentBox.startX, this.currentBox.endX),
            startY: Math.min(this.currentBox.startY, this.currentBox.endY),
            endY:   Math.max(this.currentBox.startY, this.currentBox.endY)
        };
        
        const minSize = 10;
        if (width < minSize || height < minSize) {
            this.annotationCanvas.removeChild(this.currentBox.element);
        } else {
            this.selectionBoxes.push(this.currentBox);
            this.addBoxHandles(this.currentBox);
            this.updateSelectionInfo();
        }
        
        this.updateBoxPosition(this.currentBox);

        this.currentBox = null;
        this.checkSubmitReady();
    }
    
    createBoxElement() {
        const box = document.createElement('div');
        box.className = 'selection-box';
        return box;
    }
    
    addBoxHandles(box) {
        const boxIndex = this.selectionBoxes.length;
        box.confidence = null;
        
        // Create label container with group name and confidence buttons inline
        const label = document.createElement('div');
        label.className = 'box-label';
        
        const groupText = document.createElement('span');
        groupText.textContent = `Group ${boxIndex}`;
        label.appendChild(groupText);
        
        // Add confidence buttons inline
        const confidenceContainer = document.createElement('div');
        confidenceContainer.className = 'box-confidence-container';
        
        for (let i = 1; i <= 5; i++) {
            const btn = document.createElement('button');
            btn.className = 'box-confidence-btn';
            btn.textContent = i;
            btn.dataset.rating = i;
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.setBoxRating(box.id, i);
            });
            confidenceContainer.appendChild(btn);
        }
        
        label.appendChild(confidenceContainer);
        box.element.appendChild(label);
        box.labelElement = label;
        box.confidenceContainer = confidenceContainer;
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'box-delete';
        deleteBtn.textContent = '×';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            this.deleteBox(box);
        };
        box.element.appendChild(deleteBtn);
        
        const handles = ['tl', 'tr', 'bl', 'br'];
        box.handles = {};
        
        handles.forEach(h => {
            const handle = document.createElement('div');
            handle.className = `box-handle handle-${h}`;
            handle.dataset.handle = h;
            handle.dataset.boxId = box.id;
            box.handles[h] = handle;
            box.element.appendChild(handle);
        });
    }
    
    updateBoxPosition(box) {
        const rect = this.annotationCanvas.getBoundingClientRect();
        const imgRect = this.annotationImage.getBoundingClientRect();
        
        const x1 = Math.min(box.startX, box.endX);
        const y1 = Math.min(box.startY, box.endY);
        const x2 = Math.max(box.startX, box.endX);
        const y2 = Math.max(box.startY, box.endY);
        
        const displayX1 = (x1 / this.NORMALIZED_WIDTH) * imgRect.width;
        const displayY1 = (y1 / this.NORMALIZED_HEIGHT) * imgRect.height;
        const displayX2 = (x2 / this.NORMALIZED_WIDTH) * imgRect.width;
        const displayY2 = (y2 / this.NORMALIZED_HEIGHT) * imgRect.height;
        
        const canvasX = imgRect.left - rect.left + displayX1;
        const canvasY = imgRect.top - rect.top + displayY1;
        const width = displayX2 - displayX1;
        const height = displayY2 - displayY1;
        
        box.element.style.left = `${canvasX}px`;
        box.element.style.top = `${canvasY}px`;
        box.element.style.width = `${width}px`;
        box.element.style.height = `${height}px`;
        
        // Smart label positioning - flip to left if it would overflow right edge
        if (box.labelElement) {
            const labelWidth = box.labelElement.offsetWidth || 200; // Approximate width
            const boxLeft = canvasX;
            const canvasWidth = rect.width;

            const deleteBtn = box.element.querySelector('.box-delete');
            const boxRight = canvasX + width;

            // Check if label would overflow the right edge
            if (boxLeft + labelWidth > canvasWidth) {
                box.labelElement.style.left = 'auto';
                box.labelElement.style.right = '0px';
                deleteBtn.style.left = 'auto';
                deleteBtn.style.right = '-15px';
            } else {
                box.labelElement.style.left = '0px';
                box.labelElement.style.right = 'auto';
                deleteBtn.style.left = '-15px';
                deleteBtn.style.right = 'auto';
            }
        }
    }
    
    repositionAllBoxes() {
        this.selectionBoxes.forEach(box => this.updateBoxPosition(box));
    }
    
    startResize(e) {
        this.isResizing = true;
        this.resizeHandle = e.target.dataset.handle;
        const boxId = parseInt(e.target.dataset.boxId);
        
        this.currentBox = this.selectionBoxes.find(box => box.id === boxId);
    }
    
    updateResize(e) {
        if (!this.isResizing || !this.currentBox) return;
        
        const imgRect = this.annotationImage.getBoundingClientRect();
        const pixelX = e.clientX - imgRect.left;
        const pixelY = e.clientY - imgRect.top;
        
        const normalizedX = (pixelX / imgRect.width) * this.NORMALIZED_WIDTH;
        const normalizedY = (pixelY / imgRect.height) * this.NORMALIZED_HEIGHT;
        
        if (this.resizeHandle === 'tl') {
            this.currentBox.startX = normalizedX;
            this.currentBox.startY = normalizedY;
        } else if (this.resizeHandle === 'tr') {
            this.currentBox.endX = normalizedX;
            this.currentBox.startY = normalizedY;
        } else if (this.resizeHandle === 'bl') {
            this.currentBox.startX = normalizedX;
            this.currentBox.endY = normalizedY;
        } else if (this.resizeHandle === 'br') {
            this.currentBox.endX = normalizedX;
            this.currentBox.endY = normalizedY;
        }

        this.updateBoxPosition(this.currentBox);
    }
    
    endResize() {
        this.isResizing = false;
        this.resizeHandle = null;
        this.updateSelectionInfo();
        this.currentBox = null;
    }
    
    deleteBox(box) {
        this.annotationCanvas.removeChild(box.element);
        const index = this.selectionBoxes.indexOf(box);
        if (index > -1) {
            this.selectionBoxes.splice(index, 1);
        }
        
        this.updateBoxLabels();
        this.updateSelectionInfo();
        this.checkSubmitReady();
    }
    
    updateBoxLabels() {
        this.selectionBoxes.forEach((box, i) => {
            if (box.labelElement) {
                const groupText = box.labelElement.querySelector('span');
                if (groupText) {
                    groupText.textContent = `Group ${i + 1}`;
                }
            }
        });
    }
    
    clearAllBoxes() {
        while (this.selectionBoxes.length > 0) {
            const box = this.selectionBoxes[0];
            this.annotationCanvas.removeChild(box.element);
            this.selectionBoxes.shift();
        }
        this.updateSelectionInfo();
    }
    
    updateSelectionInfo() {
        this.boxCount.textContent = this.selectionBoxes.length;
        
        if (this.selectionBoxes.length === 0) {
            this.selectionCoords.innerHTML = 'Click and drag on the image to draw boxes';
            return;
        }
        
        let html = '';
        this.selectionBoxes.forEach((box, i) => {
            const x1 = Math.round(Math.min(box.startX, box.endX));
            const y1 = Math.round(Math.min(box.startY, box.endY));
            const x2 = Math.round(Math.max(box.startX, box.endX));
            const y2 = Math.round(Math.max(box.startY, box.endY));
            
            const confidenceText = box.confidence ? `Confidence: ${box.confidence}` : 'No confidence set';
            
            html += `<div class="box-item">
                <strong>Group ${i + 1}</strong><br>
                Coordinates: (${x1}, ${y1}, ${x2}, ${y2})<br>
                <span style="color: ${box.confidence ? '#4CAF50' : '#ff9800'};">${confidenceText}</span>
            </div>`;
        });
        
        this.selectionCoords.innerHTML = html;
    }

    setBoxRating(boxId, rating) {
        const box = this.selectionBoxes.find(b => b.id === boxId);
        if (box) {
            box.confidence = rating;

            // Update button styles
            if (box.confidenceContainer) {
                const buttons = box.confidenceContainer.querySelectorAll('.box-confidence-btn');
                buttons.forEach(btn => {
                    if (parseInt(btn.dataset.rating) === rating) {
                        btn.classList.add('selected');
                    } else {
                        btn.classList.remove('selected');
                    }
                });
            }

            // Make the box body not block clicks, but keep handles and buttons interactive
            box.element.style.pointerEvents = 'none';

            // Re-enable pointer events for interactive elements
            if (box.labelElement) box.labelElement.style.pointerEvents = 'auto';
            if (box.confidenceContainer) box.confidenceContainer.style.pointerEvents = 'auto';
            const deleteBtn = box.element.querySelector('.box-delete');
            if (deleteBtn) deleteBtn.style.pointerEvents = 'auto';
            Object.values(box.handles).forEach(handle => {
                handle.style.pointerEvents = 'auto';
            });

            // Ensure unrated boxes remain fully interactive
            this.selectionBoxes.forEach(b => {
                if (b.id !== boxId && (b.confidence === null || b.confidence === undefined)) {
                    b.element.style.pointerEvents = 'auto';
                }
            });

            this.updateSelectionInfo();
        }

        this.checkSubmitReady();
    }

    setPlaybackSpeed(speed) {
        this.playbackSpeed = speed;
        
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
        this.currentFrame = Math.min(this.currentFrame + 1, this.totalFrames - 1);
        this.loadVideoFrame(this.getCurrentVideo().index, this.currentFrame);
        this.updateDisplay();
        this.videoScrubber.value = this.currentFrame;
    }
    
    previousFrame() {
        if (this.isPlaying) {
            this.pause();
        }
        this.currentFrame = Math.max(this.currentFrame - 1, 0);
        this.loadVideoFrame(this.getCurrentVideo().index, this.currentFrame);
        this.updateDisplay();
        this.videoScrubber.value = this.currentFrame;
    }
    
    skipFrames(count) {
        if (this.isPlaying) {
            this.pause();
        }
        this.currentFrame = Math.max(0, Math.min(this.currentFrame + count, this.totalFrames - 1));
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
    
    checkSubmitReady() {
        if (this.selectionBoxes.length === 0) {
            this.submitBtn.disabled = false;
            return;
        }
        
        const allRated = this.selectionBoxes.every(box => box.confidence !== null && box.confidence !== undefined);
        this.submitBtn.disabled = !allRated;
    }
    
    async submitAnnotation() {
        if (this.selectionBoxes.length > 0) {
            const allRated = this.selectionBoxes.every(box => box.confidence !== null && box.confidence !== undefined);
            if (!allRated) {
                alert('Please rate all groups before submitting, or delete boxes to skip this video');
                return;
            }
        }
        
        if (this.isPlaying) {
            this.pause();
        }
        
        const boxes = this.selectionBoxes.map((box, i) => {
            const x1 = Math.round(Math.min(box.startX, box.endX));
            const y1 = Math.round(Math.min(box.startY, box.endY));
            const x2 = Math.round(Math.max(box.startX, box.endX));
            const y2 = Math.round(Math.max(box.startY, box.endY));
            
            return {
                groupId: i + 1,
                bbox: [x1, y1, x2, y2],
                confidence: box.confidence
            };
        });
        
        const video = this.getCurrentVideo();
        const annotation = {
            timestamp: new Date().toISOString(),
            videoIndex: this.currentVideoIndex + 1,
            globalIndex: this.globalVideoIndex,
            videoFolder: video.folder,
            videoWatched: this.hasWatchedVideo,
            totalWatchTimeMs: this.totalWatchTime + (this.videoPlayStartTime ? (Date.now() - this.videoPlayStartTime) : 0),
            numberOfGroups: this.selectionBoxes.length,
            groups: boxes,
            videoInfo: {
                totalFrames: this.totalFrames,
                annotationFrame: 1,
                coordinateSystem: 'normalized',
                normalizedWidth: this.NORMALIZED_WIDTH,
                normalizedHeight: this.NORMALIZED_HEIGHT
            }
        };
        
        try {
            await this.saveAnnotation(annotation)
            if (this.currentVideoIndex < this.totalVideos - 1) {
                this.loadVideo(this.currentVideoIndex + 1);
            } else {
                window.location.href = "/thank_you";
            }
        } catch (error) {
            console.error('Error saving annotation:', error);
            alert('Error saving annotation. Please check the console.');
        }
    }

    async saveAnnotation(annotation){
        try{
            await fetch('/api/save-annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(annotation)
            });
        } catch (error){
            console.error('Error saving annotation:', error);
        }
    }
    
    async saveAllAnnotations() {
        const finalData = {
            completedAt: new Date().toISOString(),
            totalVideos: this.totalVideos,
            annotations: this.allAnnotations
        };
        
        try {
            await fetch('/api/save-all-annotations', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(finalData)
            });
        } catch (error) {
            console.error('Error saving all annotations:', error);
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
        this.checkSubmitReady();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VideoAnnotationTool();
});
