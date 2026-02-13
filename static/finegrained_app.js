/**
 * Fine-Grained Person Annotation Tool
 * Based on advisor's annotation tool architecture
 */

class VideoAnnotationTool {
    constructor() {
        this.videos = [];
        this.currentVideoIndex = 0;
        this.annotatorId = null;
        this.boxes = [];
        this.coarseGroups = [];
        this.addBoxMode = false;
        this.isDrawing = false;
        this.currentDrawStart = null;
        this.tempRect = null;
        this.selectedBox = null;
        this.selectedGroup = null;  // Track selected group for display filtering
        this.expandedGroups = new Set();  // Track which groups are expanded
        this._expandedGroupsBackup = null; // backup for restore when toggling selection/showAll
        this.expandOnDetect = true; // whether detect() should auto-expand the first group
        this.showAll = false;
        this.videoPlaying = false;
        this.currentFrame = 0;
        this.totalFrames = 50;
        this.startTime = Date.now();
        this.videoStartTime = 0;
        this.allAnnotations = [];
        this.annotatedFrame = 1;
        
        // Tracking fields (mirrors app.js patterns)
        this.videoActionHistory = [];
        this.sessionStartTime = new Date().toISOString();
        this.videoLoadTime = null;
        this.totalWatchTime = 0;
        this.videoPlayStartTime = null;
        this.deletedBoxes = [];

        this.NORMALIZED_WIDTH = 1920;
        this.NORMALIZED_HEIGHT = 1080;
        
        this.annotationCanvas = document.getElementById('annotationCanvas');
        this.annotationImage = document.getElementById('annotationImage');
        this.videoImage = document.getElementById('videoImage');
        this.playBtn = document.getElementById('playBtn');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.submitBtn = document.getElementById('submitBtn');
        this.videoScrubber = document.getElementById('videoScrubber');
        this.currentFrameNum = document.getElementById('currentFrameNum');
        this.videoFolderName = document.getElementById('videoFolderName');
        
        this.init();
    }
    
    async init() {
        // Get or create annotator ID
        const res = await fetch('/api/get-annotator-id');
        const data = await res.json();
        this.annotatorId = data.annotator_id;
        
        // Load videos
        await this.loadVideos();
        this.setupEventListeners();
    }
    
    // Track video actions with timestamps (mirrors app.js logVideoAction)
    logVideoAction(action, metadata = {}) {
        this.videoActionHistory.push({
            timestamp: new Date().toISOString(),
            timeSinceVideoLoad: this.videoLoadTime
                ? Date.now() - new Date(this.videoLoadTime).getTime()
                : 0,
            action,
            currentFrame: this.currentFrame,
            isPlaying: this.videoPlaying,
            ...metadata
        });
    }
    
    async loadVideos() {
        try {
            const res = await fetch('/api_finegrained/detect-videos');
            const data = await res.json();
            
            if (data.success) {
                this.videos = data.videos;
                this.annotatedFrame = data.annotatedFrame;

                document.getElementById('totalVideoNum').textContent = this.videos.length;
                
                // Load first video
                if (this.videos.length > 0) {
                    await this.loadVideo(0);
                }
            }
        } catch (error) {
            console.error('Error loading videos:', error);
            this.showError('FaionCanvasMouseDownled to load videos');
        }
    }
    
    async loadVideo(index) {
        this.currentVideoIndex = index;
        this.boxes = [];
        this.coarseGroups = [];
        this.selectedBox = null;
        this.selectedGroup = null;  // Reset selected group when loading new video
        this.expandedGroups = new Set();  // Reset expanded groups
        this.showAll = false;
        this.currentFrame = 0;
        this.videoPlaying = false;
        this.showCoarseGroups = false;
        this.currentFrameNum.textContent = this.annotatedFrame;
        
        // Reset per-video tracking state
        this.videoActionHistory = [];
        this.videoLoadTime = new Date().toISOString();
        this.totalWatchTime = 0;
        this.videoPlayStartTime = null;
        this.deletedBoxes = [];
        
        const video = this.videos[index];

        this.videoFolderName.textContent = video.folder;
        this.logVideoAction('video_loaded', {
            videoIndex: index,
            videoFolder: video.folder,
            annotatedFrame: this.annotatedFrame
        });
        
        // Reset UI
        document.getElementById('toggleCoarseBtn').textContent = '📦 Show Coarse Groups';
        document.getElementById('toggleCoarseBtn').style.opacity = '1';
        
        // Clear previous detections from canvas
        const svg = document.getElementById('canvasOverlay');
        svg.innerHTML = '';
        
        document.getElementById('currentVideoNum').textContent = index + 1;
        document.getElementById('boxCount').textContent = '0';
        document.getElementById('selectionCoords').innerHTML = 'Loading frames...';
        
        // Load coarse groups first
        try {
            const res = await fetch(`/api_finegrained/coarse-groups/${video.index}/frame/${this.annotatedFrame}`);
            const data = await res.json();
            this.coarseGroups = data.groups || [];
        } catch (error) {
            console.error('Error loading coarse groups:', error);
        }
        
        // Setup image load/error handlers BEFORE setting src
        this.annotationImage.onerror = () => {
            console.error('Failed to load annotation frame image');
            this.showError(`Failed to load annotation frame: frame ${this.annotatedFrame}`);
            document.getElementById('selectionCoords').innerHTML = 'Failed to load frame';
        };
        
        this.videoImage.onerror = () => {
            console.error('Failed to load video frame');
        };
        
        // Load annotation frame
        const img_frameUrl = `/api_finegrained/video/${video.index}/frame/${this.annotatedFrame}`;
        console.log(`Loading annotation frame: ${img_frameUrl}`);
        this.annotationImage.src = img_frameUrl;
        
        // Reset video to frame 0
        const frameUrl = `/api_finegrained/video/${video.index}/frame/0`;
        console.log(`Loading video frame: ${frameUrl}`);
        this.videoImage.src = frameUrl;
        document.getElementById('frameCounter').textContent = '0'; // 0-based
        document.getElementById('videoScrubber').value = '0';
        document.getElementById('totalFrames').textContent = '49';
        
        // Auto-run detection as soon as the annotation frame is loaded
        this.annotationImage.onload = () => {
            console.log('Annotation frame loaded, starting detection...');
            this.detect();
        };
    }
    
    setupEventListeners() {
        this.submitBtn.addEventListener('click', () => this.submitVideo());
        
        // Toggle coarse groups
        document.getElementById('toggleCoarseBtn').addEventListener('click', () => this.toggleCoarseGroups());
        const showAllBtn = document.getElementById('toggleShowAllBtn');
        if (showAllBtn) {
            showAllBtn.addEventListener('click', () => this.toggleShowAll());
        }

        const addBoxBtn = document.getElementById('toggleAddBoxBtn');
        if (addBoxBtn) {
            addBoxBtn.addEventListener('click', () => this.toggleAddBox());
        }
        
        this.playBtn.addEventListener('click', () => this.playVideo());
        this.pauseBtn.addEventListener('click', () => this.pauseVideo());
        
        this.videoScrubber.addEventListener('input', (e) => {
            const previousFrame = this.currentFrame;
            const newFrame = parseInt(e.target.value);
            this.logVideoAction('scrubber_seek', {
                fromFrame: previousFrame,
                toFrame: newFrame,
                frameDelta: newFrame - previousFrame
            });
            this.currentFrame = newFrame;
            document.getElementById('frameCounter').textContent = this.currentFrame;
            this.updateVideoFrame();
        });
        
        // Click on annotation canvas to select boxes or draw new ones when Add Box enabled
        this.annotationCanvas.addEventListener('mousedown', (e) => this.onCanvasMouseDown(e));
        this.annotationCanvas.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
        this.annotationCanvas.addEventListener('mouseup', () => this.onCanvasMouseUp());
        this.annotationCanvas.addEventListener('mouseleave', () => this.onCanvasMouseUp());
        
        // Double-click on SVG overlay to edit
        const svg = document.getElementById('canvasOverlay');
        svg.addEventListener('dblclick', (e) => this.onCanvasDoubleClick(e));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        
        // Window resize handler to redraw boxes
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.drawBoxes();
            }, 100); // Debounce to avoid too many redraws
        });
        
        // Also redraw when annotation image loads/changes size
        this.annotationImage.addEventListener('load', () => {
            // Wait a bit for the image to fully render and get its final size
            setTimeout(() => this.drawBoxes(), 50);
        });
    }
    
    async detect() {
        document.getElementById('loadingSpinner').style.display = 'block';
        
        try {
            const video = this.videos[this.currentVideoIndex];
            const res = await fetch(`/api_finegrained/detect-frame/${video.index}/frame/${this.annotatedFrame}`);
            const data = await res.json();
            
            if (data.success) {
                this.boxes = data.boxes;
                this.coarseGroups = data.coarseGroups;
                // Preserve backend-assigned group numbers; attach numeric groupId for convenience
                this.boxes.forEach(b => {
                    if (b.label && b.label.startsWith('group_')) {
                        const m = b.label.match(/(\d+)/);
                        b.groupId = m ? parseInt(m[1], 10) : undefined;
                    } else {
                        b.groupId = undefined;
                    }
                });
                document.getElementById('boxCount').textContent = this.boxes.length;
                // Auto-select first group for display (only if enabled)
                const firstGroup = this.getFirstGroupLabel();
                if (this.expandOnDetect && firstGroup) {
                    this.selectedGroup = firstGroup;
                    this.expandedGroups.add(firstGroup);
                }
                // Reset flag so future detects behave normally
                this.expandOnDetect = true;
                this.updateGroupDisplay();
                this.drawBoxes();
            } else {
                this.showError('Detection failed: ' + data.error);
            }
        } catch (error) {
            console.error('Error during detection:', error);
            this.showError('Detection error: ' + error.message);
        } finally {
            document.getElementById('loadingSpinner').style.display = 'none';
        }
    }
    
    drawBoxes() {
        // Clear canvas
        const svg = document.getElementById('canvasOverlay');
        svg.innerHTML = '';
        
        if (!this.annotationImage.complete || !this.annotationImage.naturalHeight) {
            return;
        }
        
        const container = this.annotationCanvas;
        const imgRect = this.annotationImage.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        
        // Calculate offset
        const offsetX = imgRect.left - containerRect.left;
        const offsetY = imgRect.top - containerRect.top;
        const scale = imgRect.width / this.annotationImage.naturalWidth;
        
        const activeGroupLabels = this.getActiveGroupLabels();
        // Build list of boxes to draw with original indices to keep selection mapping stable
        const boxesToDraw = [];

        // Determine which boxes to show:
        // - If showAll: show everything
        // - Otherwise, show only boxes from expanded groups (activeGroupLabels)
        // - If a person is selected and their group is NOT expanded, show only that single person
        const showEverything = this.showAll;
        this.boxes.forEach((box, originalIdx) => {
            if (showEverything) {
                boxesToDraw.push({ box, originalIdx });
                return;
            }

            // Default: only show boxes from expanded groups. If no groups expanded, show none.
            if (activeGroupLabels.size === 0) return;
            if (activeGroupLabels.has(box.label)) {
                boxesToDraw.push({ box, originalIdx });
            }
        });
        
        // Draw detection boxes FIRST (so they're on top for interaction)
        // Assign person ids per group for labeling
        const groupCounters = {};
        boxesToDraw.forEach(({ box }, _i) => {
            const key = box.label || 'individual';
            if (!groupCounters[key]) groupCounters[key] = 0;
        });

        // Separate boxes: non-selected first, selected last (ensures selected box is on top)
        const nonSelectedBoxes = [];
        let selectedBoxData = null;
        
        boxesToDraw.forEach(({ box, originalIdx }) => {
            if (this.selectedBox === originalIdx) {
                selectedBoxData = { box, originalIdx };
            } else {
                nonSelectedBoxes.push({ box, originalIdx });
            }
        });
        
        // Draw all non-selected boxes first
        [...nonSelectedBoxes, ...(selectedBoxData ? [selectedBoxData] : [])].forEach(({ box, originalIdx }) => {
            const x = box.tl_x * scale + offsetX;
            const y = box.tl_y * scale + offsetY;
            const width = (box.br_x - box.tl_x) * scale;
            const height = (box.br_y - box.tl_y) * scale;
            
            const isSelected = this.selectedBox === originalIdx;


            // Visual emphasis: if any box is selected, make selected box more opaque and others more translucent
            let fillColor;
            let strokeColor;
            let strokeWidth = isSelected ? 5 : 3;
            if (this.selectedBox === null) {
                // no selection: default subtle style
                fillColor = this.hexToRgba(this.getLabelColor(box.label), 0.18);
                strokeColor = this.getLabelColor(box.label);
            } else {
                // there's a selection: selected box emphasized, others de-emphasized
                if (isSelected) {
                    fillColor = this.hexToRgba(this.getLabelColor(box.label), 0.5);
                    strokeColor = this.getLabelColor(box.label);
                    strokeWidth = 6;
                } else {
                    // less aggressive de-emphasis so unselected boxes remain visible
                    // Make 'individual' boxes slightly more visible (thicker border)
                    if ((box.label || 'individual') === 'individual') {
                        fillColor = this.hexToRgba(this.getLabelColor(box.label), 0.18);
                        strokeColor = this.hexToRgba(this.getLabelColor(box.label), 0.85);
                        strokeWidth = 4;
                    } else {
                        fillColor = this.hexToRgba(this.getLabelColor(box.label), 0.12);
                        strokeColor = this.hexToRgba(this.getLabelColor(box.label), 0.65);
                        strokeWidth = 3;
                    }
                }
            }

            // Draw rectangle
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x);
            rect.setAttribute('y', y);
            rect.setAttribute('width', width);
            rect.setAttribute('height', height);
            rect.setAttribute('fill', fillColor);
            rect.setAttribute('stroke', strokeColor);
            rect.setAttribute('stroke-width', strokeWidth.toString());
            rect.setAttribute('data-idx', originalIdx);
            rect.setAttribute('pointer-events', 'all');
            rect.style.cursor = 'pointer';
            
            // Add dblclick handler directly to rect: toggle selection (double-click cancels selection)
            rect.addEventListener('dblclick', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (this.selectedBox === originalIdx) {
                    this.selectedBox = null;
                } else {
                    this.selectedBox = originalIdx;
                }
                this.updateGroupDisplay();
                this.drawBoxes();
            });
            
            svg.appendChild(rect);
            
            // Draw label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x + 5);
            text.setAttribute('y', y - 5);
            // Label color: keep selected label vivid, dim others when a selection exists
            if (this.selectedBox === null) {
                text.setAttribute('fill', this.getLabelColor(box.label));
            } else {
                text.setAttribute('fill', isSelected ? this.getLabelColor(box.label) : this.hexToRgba(this.getLabelColor(box.label), 0.75));
            }
            text.setAttribute('font-size', isSelected ? '16px' : '14px');
            text.setAttribute('font-weight', '700');
            text.setAttribute('pointer-events', 'all');
            // Compute person id within its group for display
            const gkey = box.label || 'individual';
            groupCounters[gkey] = (groupCounters[gkey] || 0) + 1;
            const personId = groupCounters[gkey];
            let displayLabel = '';
            if (gkey === 'individual') {
                displayLabel = `individual_p${personId}`;
            } else {
                displayLabel = `${gkey}_p${personId}`; // e.g. group_2_p1
            }
            text.textContent = displayLabel;
            
            // Add dblclick handler to text as well: toggle selection
            text.addEventListener('dblclick', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (this.selectedBox === originalIdx) {
                    this.selectedBox = null;
                } else {
                    this.selectedBox = originalIdx;
                }
                this.updateGroupDisplay();
                this.drawBoxes();
            });
            
            svg.appendChild(text);
            
            // Add resize handles if selected
            if (isSelected) {
                const handleSize = 8;
                const handleColor = this.getLabelColor(box.label);
                const corners = [
                    { pos: 'tl', cx: x, cy: y },
                    { pos: 'tr', cx: x + width, cy: y },
                    { pos: 'bl', cx: x, cy: y + height },
                    { pos: 'br', cx: x + width, cy: y + height }
                ];
                
                    corners.forEach((corner, idx) => {
                        const handle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                        handle.setAttribute('cx', corner.cx);
                        handle.setAttribute('cy', corner.cy);
                        handle.setAttribute('r', handleSize);
                        handle.setAttribute('fill', handleColor);
                        handle.setAttribute('stroke', 'white');
                        handle.setAttribute('stroke-width', '2');
                        handle.setAttribute('data-idx', originalIdx);
                        handle.setAttribute('data-corner', corner.pos);
                        handle.setAttribute('cursor', 'grab');
                        handle.style.cursor = 'grab';
                    
                        handle.addEventListener('mousedown', (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            this.startResizeBox(e, originalIdx, corner.pos, scale, offsetX, offsetY);
                        });
                    
                        svg.appendChild(handle);
                    });
                
                // Add Edit and Delete buttons for selected box
                const buttonY = y - 35;
                const buttonSize = 26;
                const buttonSpacing = 5;
                
                // Delete button (X) - left position
                const deleteButtonX = x;
                const deleteBtn = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                deleteBtn.style.cursor = 'pointer';
                
                const deleteBg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                deleteBg.setAttribute('cx', deleteButtonX + buttonSize / 2);
                deleteBg.setAttribute('cy', buttonY + buttonSize / 2);
                deleteBg.setAttribute('r', buttonSize / 2);
                deleteBg.setAttribute('fill', '#ff4444');
                deleteBtn.appendChild(deleteBg);
                
                const deleteText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                deleteText.setAttribute('x', deleteButtonX + buttonSize / 2);
                deleteText.setAttribute('y', buttonY + buttonSize / 2 + 6);
                deleteText.setAttribute('fill', 'white');
                deleteText.setAttribute('font-size', '18px');
                deleteText.setAttribute('font-weight', 'bold');
                deleteText.setAttribute('text-anchor', 'middle');
                deleteText.setAttribute('pointer-events', 'all');
                deleteText.textContent = '×';
                deleteBtn.appendChild(deleteText);
                
                deleteBtn.addEventListener('mouseenter', () => {
                    deleteBg.setAttribute('fill', '#cc0000');
                });
                deleteBtn.addEventListener('mouseleave', () => {
                    deleteBg.setAttribute('fill', '#ff4444');
                });
                deleteBtn.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.deleteBox(originalIdx);
                });
                
                svg.appendChild(deleteBtn);
                
                // Edit button - right position
                const editButtonX = deleteButtonX + buttonSize + buttonSpacing;
                const editBtn = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                editBtn.style.cursor = 'pointer';
                
                const editBg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                editBg.setAttribute('cx', editButtonX + buttonSize / 2);
                editBg.setAttribute('cy', buttonY + buttonSize / 2);
                editBg.setAttribute('r', buttonSize / 2);
                editBg.setAttribute('fill', this.getLabelColor(box.label));
                editBtn.appendChild(editBg);
                
                const editText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                editText.setAttribute('x', editButtonX + buttonSize / 2);
                editText.setAttribute('y', buttonY + buttonSize / 2 + 6);
                editText.setAttribute('fill', 'white');
                editText.setAttribute('font-size', '16px');
                editText.setAttribute('font-weight', 'bold');
                editText.setAttribute('text-anchor', 'middle');
                editText.setAttribute('pointer-events', 'all');
                editText.textContent = '✎';
                editBtn.appendChild(editText);
                
                editBtn.addEventListener('mouseenter', () => {
                    const currentColor = this.getLabelColor(box.label);
                    const darkerColor = this.darkenColor(currentColor, 20);
                    editBg.setAttribute('fill', darkerColor);
                });
                editBtn.addEventListener('mouseleave', () => {
                    editBg.setAttribute('fill', this.getLabelColor(box.label));
                });
                editBtn.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.editLabel(originalIdx);
                });
                
                svg.appendChild(editBtn);
            }
        });
        
        // Draw coarse groups LAST (so they're in background and don't interfere)
        if (this.showCoarseGroups) {
            this.coarseGroups.forEach((group, idx) => {
                const groupLabel = `group_${group.groupId}`;
                if (!this.showAll && activeGroupLabels.size > 0 && !activeGroupLabels.has(groupLabel)) {
                    return;
                }
                const x = group.bbox[0] * scale + offsetX;
                const y = group.bbox[1] * scale + offsetY;
                const width = (group.bbox[2] - group.bbox[0]) * scale;
                const height = (group.bbox[3] - group.bbox[1]) * scale;
                
                // Semi-transparent background for visibility
                const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                bgRect.setAttribute('x', x);
                bgRect.setAttribute('y', y);
                bgRect.setAttribute('width', width);
                bgRect.setAttribute('height', height);
                bgRect.setAttribute('fill', 'rgba(100, 149, 237, 0.15)');
                bgRect.setAttribute('stroke', 'none');
                bgRect.setAttribute('pointer-events', 'none');
                svg.appendChild(bgRect);
                
                // Prominent border
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y);
                rect.setAttribute('width', width);
                rect.setAttribute('height', height);
                rect.setAttribute('fill', 'none');
                rect.setAttribute('stroke', '#4A90E2');
                rect.setAttribute('stroke-width', '3');
                rect.setAttribute('stroke-dasharray', '8,4');
                rect.setAttribute('pointer-events', 'none');
                svg.appendChild(rect);
                
                // Label on top-left corner
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x + 5);
                text.setAttribute('y', y - 5);
                text.setAttribute('fill', '#4A90E2');
                text.setAttribute('font-size', '12px');
                text.setAttribute('font-weight', 'bold');
                text.setAttribute('background', 'rgba(255,255,255,0.8)');
                text.setAttribute('pointer-events', 'none');
                text.textContent = `Group ${group.groupId}`;
                svg.appendChild(text);
            });
        }
    }
    
    startResizeBox(e, boxIdx, corner, scale, offsetX, offsetY) {
        const box = this.boxes[boxIdx];
        let startX = e.clientX;
        let startY = e.clientY;
        let lastX = box.tl_x;
        let lastY = box.tl_y;
        let lastBrX = box.br_x;
        let lastBrY = box.br_y;
        
            const onMouseMove = (moveEvent) => {
            const currentX = moveEvent.clientX;
            const currentY = moveEvent.clientY;
            
            const deltaX = (currentX - startX) / scale;
            const deltaY = (currentY - startY) / scale;
            
            // Reset to last saved state
            box.tl_x = lastX;
            box.tl_y = lastY;
            box.br_x = lastBrX;
            box.br_y = lastBrY;
            
            // Apply delta based on corner
            if (corner === 'tl') {
                box.tl_x += deltaX;
                box.tl_y += deltaY;
            } else if (corner === 'tr') {
                box.br_x += deltaX;
                box.tl_y += deltaY;
            } else if (corner === 'bl') {
                box.tl_x += deltaX;
                box.br_y += deltaY;
            } else if (corner === 'br') {
                box.br_x += deltaX;
                box.br_y += deltaY;
            }
            
            // Ensure min size (10 pixels)
            if (box.br_x - box.tl_x < 10) {
                if (corner === 'tl' || corner === 'bl') {
                    box.tl_x = box.br_x - 10;
                } else {
                    box.br_x = box.tl_x + 10;
                }
            }
            if (box.br_y - box.tl_y < 10) {
                if (corner === 'tl' || corner === 'tr') {
                    box.tl_y = box.br_y - 10;
                } else {
                    box.br_y = box.tl_y + 10;
                }
            }
            
            // Clamp to normalized image bounds so handles never disappear off-canvas
            const MIN_X = 0;
            const MIN_Y = 0;
            const MAX_X = this.NORMALIZED_WIDTH;
            const MAX_Y = this.NORMALIZED_HEIGHT;

            box.tl_x = Math.max(MIN_X, Math.min(box.tl_x, MAX_X));
            box.tl_y = Math.max(MIN_Y, Math.min(box.tl_y, MAX_Y));
            box.br_x = Math.max(MIN_X, Math.min(box.br_x, MAX_X));
            box.br_y = Math.max(MIN_Y, Math.min(box.br_y, MAX_Y));

            // Ensure proper ordering after clamping
            if (box.br_x <= box.tl_x) box.br_x = box.tl_x + 10;
            if (box.br_y <= box.tl_y) box.br_y = box.tl_y + 10;

            this.drawBoxes();
            // Update the right-side list to reflect coordinate changes live
            this.updateGroupDisplay();
        };
        
        const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            this.updateGroupDisplay();
        };
        
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    }
    getLabelColor(label) {
        // Vibrant color palette with high contrast for different groups
        const colors = [
            '#e6194b', '#f58231', '#ffe119', '#bfef45', '#3cb44b',
            '#a9a9a9', '#911eb4', '#f032e6', '#fabed4', '#9a6324',
            '#800000', '#ffd8b1', '#808000', '#fffac8', '#4363d8',
            '#f4a261', '#e76f51', '#2a9d8f', '#8ac926', '#ffca3a',
            '#ff595e', '#ff924c', '#c77dff', '#7209b7', '#b5179e',
            '#ff006e', '#d62828', '#f77f00', '#fcbf49', '#6a994e'
        ];
        
        if (label === 'individual') {
            return '#999999';
        }

        const match = label.match(/group_(\d+)/);
        if (match) {
            const id = parseInt(match[1], 10);
            // Use golden angle to pick a hue that spreads colors evenly
            const goldenAngle = 137.508; // degrees
            const hue = (id * goldenAngle) % 360;
            // Vary saturation and lightness deterministically by id to increase contrast
            const saturation = 60 + ((id * 37) % 25); // 60-84%
            const lightness = 45 + ((id * 29) % 20); // 45-64%

            // Convert HSL to RGB then to hex
            function hslToRgb(h, s, l) {
                s /= 100;
                l /= 100;
                const k = n => (n + h / 30) % 12;
                const a = s * Math.min(l, 1 - l);
                const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
                return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
            }

            function rgbToHex(r, g, b) {
                return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
            }

            const [r, g, b] = hslToRgb(hue, saturation, lightness);
            const hex = rgbToHex(r, g, b);

            // Occasionally assign a warm color (yellow/orange) for better visual variety
            const warmPalette = ['#ffba08', '#ff6b00', '#ff8f00', '#ffd166', '#ffb703'];
            if (id % 6 === 0) {
                return warmPalette[id % warmPalette.length];
            }
            // Avoid returning same color as individual or coarse-group
            const forbidden = new Set(['#999999', '#4A90E2']);
            if (forbidden.has(hex)) {
                // shift hue slightly
                const [r2, g2, b2] = hslToRgb((hue + 47) % 360, saturation, lightness);
                return rgbToHex(r2, g2, b2);
            }
            return hex;
        }

        return '#CCCCCC';
    }

    // Convert hex color to rgba string with given alpha (0.0 - 1.0)
    hexToRgba(hex, alpha) {
        if (!hex) return `rgba(0,0,0,${alpha})`;
        const h = hex.replace('#', '');
        const bigint = parseInt(h, 16);
        const r = (bigint >> 16) & 255;
        const g = (bigint >> 8) & 255;
        const b = bigint & 255;
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    
    // Darken a hex color by a percentage
    darkenColor(hex, percent) {
        if (!hex) return '#000000';
        const h = hex.replace('#', '');
        const bigint = parseInt(h, 16);
        let r = (bigint >> 16) & 255;
        let g = (bigint >> 8) & 255;
        let b = bigint & 255;
        
        r = Math.max(0, Math.floor(r * (1 - percent / 100)));
        g = Math.max(0, Math.floor(g * (1 - percent / 100)));
        b = Math.max(0, Math.floor(b * (1 - percent / 100)));
        
        return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
    }
    
    onCanvasMouseDown(e) {
        // If Add Box mode is enabled, start drawing a new box
        if (this.addBoxMode) {
            this.startAddBox(e);
            return;
        }

        // Check if clicking on a box
        const svg = document.getElementById('canvasOverlay');
        const rect = svg.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Find clicked box
        const rects = svg.querySelectorAll('rect');
        for (const svgRect of rects) {
            const rx = parseFloat(svgRect.getAttribute('x'));
            const ry = parseFloat(svgRect.getAttribute('y'));
            const rw = parseFloat(svgRect.getAttribute('width'));
            const rh = parseFloat(svgRect.getAttribute('height'));
            
            if (x >= rx && x <= rx + rw && y >= ry && y <= ry + rh) {
                const idx = parseInt(svgRect.getAttribute('data-idx'));
                // Toggle selection if clicking the already-selected box
                if (this.selectedBox === idx) {
                    this.selectedBox = null;
                } else {
                    this.selectedBox = idx;
                }
                this.updateGroupDisplay(); // sync right-side list
                this.drawBoxes(); // Redraw to highlight
                break;
            }
        }
    }
    
    onCanvasDoubleClick(e) {
        // Prevent default behavior
        e.preventDefault();
        e.stopPropagation();
        
        // Check if double-clicking on a rect element
        if (e.target.tagName === 'rect' || e.target.tagName === 'text') {
            // Find the associated rect
            let rect = e.target;
            if (e.target.tagName === 'text') {
                // If clicking text, find the previous rect
                rect = e.target.previousElementSibling;
            }
            
            if (rect && rect.getAttribute('data-idx')) {
                const idx = parseInt(rect.getAttribute('data-idx'));
                // Toggle selection on double-click (do not open edit dialog here)
                if (this.selectedBox === idx) {
                    this.selectedBox = null;
                } else {
                    this.selectedBox = idx;
                }
                this.updateGroupDisplay();
                this.drawBoxes();
                return;
            }
        }
        
        // Fallback: check if clicking on SVG coordinates
        const svg = document.getElementById('canvasOverlay');
        const svgRect = svg.getBoundingClientRect();
        const x = e.clientX - svgRect.left;
        const y = e.clientY - svgRect.top;
        
        const rects = svg.querySelectorAll('rect');
        for (const svgElem of rects) {
            const rx = parseFloat(svgElem.getAttribute('x'));
            const ry = parseFloat(svgElem.getAttribute('y'));
            const rw = parseFloat(svgElem.getAttribute('width'));
            const rh = parseFloat(svgElem.getAttribute('height'));
            
                if (x >= rx && x <= rx + rw && y >= ry && y <= ry + rh) {
                    const idx = parseInt(svgElem.getAttribute('data-idx'));
                    if (this.selectedBox === idx) {
                        this.selectedBox = null;
                    } else {
                        this.selectedBox = idx;
                    }
                    this.updateGroupDisplay();
                    this.drawBoxes();
                    break;
                }
        }
    }
    
    onCanvasMouseMove(e) {
        // If currently drawing a new box, update it
        if (this.isDrawing) {
            this.updateAddBox(e);
            return;
        }
        // Placeholder for dragging
    }
    
    onCanvasMouseUp() {
        if (this.isDrawing) {
            this.endAddBox();
            return;
        }
        // Placeholder
    }

    toggleAddBox() {
        this.addBoxMode = !this.addBoxMode;
        const btn = document.getElementById('toggleAddBoxBtn');
        if (btn) {
            btn.style.opacity = this.addBoxMode ? '0.7' : '1';
            btn.textContent = this.addBoxMode ? '✍️ Drawing...' : '➕ Add Box';
        }
        if (!this.addBoxMode && this.isDrawing) {
            this.endAddBox();
        }
    }

    startAddBox(e) {
        const imgRect = this.annotationImage.getBoundingClientRect();
        if (!imgRect.width || !imgRect.height) return;

        const pixelX = e.clientX - imgRect.left;
        const pixelY = e.clientY - imgRect.top;

        const startX = (pixelX / imgRect.width) * this.NORMALIZED_WIDTH;
        const startY = (pixelY / imgRect.height) * this.NORMALIZED_HEIGHT;

        this.isDrawing = true;
        this.currentDrawStart = { x: startX, y: startY };

        // Create temporary SVG rect for feedback
        const svg = document.getElementById('canvasOverlay');
        this.tempRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        this.tempRect.setAttribute('x', pixelX);
        this.tempRect.setAttribute('y', pixelY);
        this.tempRect.setAttribute('width', 1);
        this.tempRect.setAttribute('height', 1);
        this.tempRect.setAttribute('fill', 'rgba(100, 149, 237, 0.15)');
        this.tempRect.setAttribute('stroke', '#4A90E2');
        this.tempRect.setAttribute('stroke-width', '2');
        svg.appendChild(this.tempRect);
    }

    updateAddBox(e) {
        if (!this.isDrawing || !this.currentDrawStart) return;
        const imgRect = this.annotationImage.getBoundingClientRect();
        const pixelX = e.clientX - imgRect.left;
        const pixelY = e.clientY - imgRect.top;

        const startPixelX = (this.currentDrawStart.x / this.NORMALIZED_WIDTH) * imgRect.width + imgRect.left - imgRect.left;
        const startPixelY = (this.currentDrawStart.y / this.NORMALIZED_HEIGHT) * imgRect.height + imgRect.top - imgRect.top;

        const x = Math.min(startPixelX, pixelX);
        const y = Math.min(startPixelY, pixelY);
        const w = Math.abs(pixelX - startPixelX);
        const h = Math.abs(pixelY - startPixelY);

        if (this.tempRect) {
            this.tempRect.setAttribute('x', x);
            this.tempRect.setAttribute('y', y);
            this.tempRect.setAttribute('width', w);
            this.tempRect.setAttribute('height', h);
        }
    }

    endAddBox() {
        if (!this.isDrawing || !this.currentDrawStart) return;
        const svg = document.getElementById('canvasOverlay');
        const imgRect = this.annotationImage.getBoundingClientRect();

        // If tempRect missing, just reset
        if (!this.tempRect) {
            this.isDrawing = false;
            this.currentDrawStart = null;
            return;
        }

        const x = parseFloat(this.tempRect.getAttribute('x'));
        const y = parseFloat(this.tempRect.getAttribute('y'));
        const w = parseFloat(this.tempRect.getAttribute('width'));
        const h = parseFloat(this.tempRect.getAttribute('height'));

        // Convert pixels back to normalized image space (1920x1080)
        const tl_x = (x / imgRect.width) * this.NORMALIZED_WIDTH;
        const tl_y = (y / imgRect.height) * this.NORMALIZED_HEIGHT;
        const br_x = ((x + w) / imgRect.width) * this.NORMALIZED_WIDTH;
        const br_y = ((y + h) / imgRect.height) * this.NORMALIZED_HEIGHT;

        // Minimum size filter
        if (Math.abs(br_x - tl_x) >= 10 && Math.abs(br_y - tl_y) >= 10) {
            // By default assign new boxes to the currently selected group (if any),
            // otherwise default to 'individual'. This allows adding to empty coarse groups.
            const assignLabel = this.selectedGroup || 'individual';
            const newBox = {
                id: `box_${Date.now()}`,
                tl_x: Math.round(tl_x),
                tl_y: Math.round(tl_y),
                br_x: Math.round(br_x),
                br_y: Math.round(br_y),
                confidence: null,
                label: assignLabel
            };
            this.boxes.push(newBox);
            // Select the newly added box and ensure its group is visible
            const newIdx = this.boxes.length - 1;
            this.selectedBox = newIdx;
            this.selectedGroup = newBox.label;
            // Ensure the group is expanded/visible after adding
            this.expandedGroups.add(newBox.label);
            this.showAll = false;
            document.getElementById('boxCount').textContent = this.boxes.length;
            this.updateGroupDisplay();
            this.drawBoxes();
        }

        // Cleanup
        if (this.tempRect && this.tempRect.parentNode) {
            this.tempRect.parentNode.removeChild(this.tempRect);
        }
        this.tempRect = null;
        this.isDrawing = false;
        this.currentDrawStart = null;
        // Exit add box mode automatically
        this.addBoxMode = false;
        const btn = document.getElementById('toggleAddBoxBtn');
        if (btn) {
            btn.style.opacity = '1';
            btn.textContent = '➕ Add Box';
        }
    }
    
    createDeleteButton(x, y) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        
        const btn = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        btn.setAttribute('cx', x + 5);
        btn.setAttribute('cy', y - 10);
        btn.setAttribute('r', '12');
        btn.setAttribute('fill', '#ff4444');
        btn.setAttribute('cursor', 'pointer');
        
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x + 5);
        text.setAttribute('y', y - 6);
        text.setAttribute('fill', 'white');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '16px');
        text.setAttribute('font-weight', 'bold');
        text.textContent = '×';
        text.setAttribute('cursor', 'pointer');
        
        btn.addEventListener('click', () => {
            if (this.selectedBox !== null) {
                this.boxes.splice(this.selectedBox, 1);
                this.selectedBox = null;
                document.getElementById('boxCount').textContent = this.boxes.length;
                this.updateGroupDisplay();
                this.drawBoxes();
            }
        });
        
        text.addEventListener('click', () => {
            if (this.selectedBox !== null) {
                this.boxes.splice(this.selectedBox, 1);
                this.selectedBox = null;
                document.getElementById('boxCount').textContent = this.boxes.length;
                this.updateGroupDisplay();
                this.drawBoxes();
            }
        });
        
        g.appendChild(btn);
        g.appendChild(text);
        return g;
    }
    
    updateGroupDisplay() {
        const list = document.getElementById('selectionCoords');
        
        if (this.boxes.length === 0) {
            // Even if there are no detected boxes, render the coarse groups (may be empty)
            // so annotators can add people into empty groups.
            const groupsFromCoarse = (this.coarseGroups || []).map(g => `group_${g.groupId}`);
            if (groupsFromCoarse.length === 0) {
                list.innerHTML = 'No detections';
                return;
            }
            // otherwise continue to render with zero counts
        }
        
        // Organize boxes by group, but start from coarseGroups so empty groups are shown
        const groups = {};
        const individuals = [];

        // Initialize groups from coarseGroups (ensures empty groups appear)
        if (Array.isArray(this.coarseGroups)) {
            this.coarseGroups.forEach(g => {
                const label = `group_${g.groupId}`;
                groups[label] = [];
            });
        }

        // Populate with actual boxes
        this.boxes.forEach((box, idx) => {
            if (box.label === 'individual' || !box.label) {
                individuals.push({ box, idx });
            } else {
                if (!groups[box.label]) {
                    groups[box.label] = [];
                }
                groups[box.label].push({ box, idx });
            }
        });
        
        // Sort group keys numerically (handles group_# labels)
        const sortedGroupKeys = Object.keys(groups).sort((a, b) => {
            const matchA = a.match(/\d+/);
            const matchB = b.match(/\d+/);
            const numA = matchA ? parseInt(matchA[0]) : Number.POSITIVE_INFINITY;
            const numB = matchB ? parseInt(matchB[0]) : Number.POSITIVE_INFINITY;
            return numA - numB;
        });
        
        // Build HTML for groups
        let html = '';
        
        // Render groups
        for (const groupLabel of sortedGroupKeys) {
            const groupBoxes = groups[groupLabel];
            const color = this.getLabelColor(groupLabel);
            const isSelected = this.selectedGroup === groupLabel;
            const isExpanded = this.expandedGroups.has(groupLabel);
            
            html += `
            <div class="group-section ${isExpanded ? 'expanded' : ''} ${isSelected ? 'selected' : ''}" data-group="${groupLabel}">
                <div class="group-header" onclick="tool.toggleGroup('${groupLabel}')">
                    <div class="group-header-left">
                        <span class="group-toggle-icon">${isExpanded ? '▼' : '▶'}</span>
                        <span style="color: ${color}; font-weight: bold;">${groupLabel}</span>
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span class="group-count">${groupBoxes.length} person${groupBoxes.length !== 1 ? 's' : ''}</span>
                        ${groupBoxes.length > 0 ? `<button onclick="event.stopPropagation(); tool.makeAllIndividual('${groupLabel}')" style="padding: 4px 8px; background: #ff9800; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: bold; white-space: nowrap;" title="Convert all persons in this group to individuals">👤 Make All Individual</button>` : ''}
                    </div>
                </div>
                <div class="group-persons" style="display: ${isExpanded ? 'block' : 'none'};">
                    ${groupBoxes.map(({ box, idx }, i) => `
                        <div class="person-item ${this.selectedBox === idx ? 'selected' : ''}" onclick="tool.selectPerson(${idx}, '${groupLabel}')">
                            <div class="person-label">
                                <span class="person-color-dot" style="background-color: ${color};"></span>
                                <span>${groupLabel === 'individual' ? `individual_p${i+1}` : `${groupLabel}_p${i+1}`}</span>
                            </div>
                            <div class="person-coords">(${Math.round(box.tl_x)}, ${Math.round(box.tl_y)}, ${Math.round(box.br_x)}, ${Math.round(box.br_y)})</div>
                            <div class="person-actions">
                                <button onclick="event.stopPropagation(); tool.editLabel(${idx})" style="background: ${color}; color: white;">Edit</button>
                                <button onclick="event.stopPropagation(); tool.deleteBox(${idx})" style="background: #ff6b6b; color: white;">×</button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            `;
        }
        
        // Render individuals
        if (individuals.length > 0) {
            const color = this.getLabelColor('individual');
            const isExpanded = this.expandedGroups.has('individual');
            html += `<div class="group-section ${this.selectedGroup === 'individual' ? 'selected' : ''}" data-group="individual">`;
            html += `<div class="group-header" onclick="tool.toggleGroup('individual')">`;
            html += `<div class="group-header-left">`;
            html += `<span class="group-toggle-icon">${isExpanded ? '▼' : '▶'}</span>`;
            html += `<span style="color: ${color}; font-weight: bold;">Individual Persons</span>`;
            html += `</div>`;
            html += `<span class="group-count">${individuals.length} person${individuals.length !== 1 ? 's' : ''}</span>`;
            html += `</div>`;
            html += `<div class="group-persons" style="display: ${isExpanded ? 'block' : 'none'};">`;
            html += individuals.map(({ box, idx }, i) => `
                <div class="person-item ${this.selectedBox === idx ? 'selected' : ''}" onclick="tool.selectPerson(${idx}, 'individual')">
                    <div class="person-label">
                        <span class="person-color-dot" style="background-color: ${color};"></span>
                        <span>individual_p${i + 1}</span>
                    </div>
                    <div class="person-coords">(${Math.round(box.tl_x)}, ${Math.round(box.tl_y)}, ${Math.round(box.br_x)}, ${Math.round(box.br_y)})</div>
                    <div class="person-actions">
                        <button onclick="event.stopPropagation(); tool.editLabel(${idx})" style="background: ${color}; color: white;">Edit</button>
                        <button onclick="event.stopPropagation(); tool.deleteBox(${idx})" style="background: #ff6b6b; color: white;">×</button>
                    </div>
                </div>
            `).join('');
            html += `</div></div>`;
        }
        
        list.innerHTML = html;
        
        // Update group count
        const totalGroups = sortedGroupKeys.length + (individuals.length > 0 ? 1 : 0);
        const groupCountEl = document.getElementById('groupCount') || document.getElementById('boxCount');
        if (groupCountEl) {
            groupCountEl.textContent = totalGroups;
        }
    }
    
    toggleGroup(groupLabel) {
        // Clicking group header selects and toggles expand
        this.selectedGroup = groupLabel;
        this.selectedBox = null;

        if (this.showAll) {
            this.showAll = false;
            this.updateShowAllButton();
        }
        
        if (this.expandedGroups.has(groupLabel)) {
            this.expandedGroups.delete(groupLabel);
        } else {
            this.expandedGroups.add(groupLabel);
        }
        this.updateGroupDisplay();
        this.drawBoxes();
    }

    toggleShowAll() {
        const turningOn = !this.showAll;
        this.showAll = turningOn;
        if (turningOn) {
            // Backup current expandedGroups so we can restore when turning off
            this._expandedGroupsBackup = new Set(this.expandedGroups);
            // Expand all groups present
            const allLabels = this.getAllGroupLabels();
            this.expandedGroups = new Set(allLabels);
            this.selectedGroup = null;
        } else {
            // Close all groups when turning off Show All (Close All)
            this.expandedGroups.clear();
            this._expandedGroupsBackup = null;
        }
        this.updateShowAllButton();
        this.updateGroupDisplay();
        this.drawBoxes();
    }

    updateShowAllButton() {
        const btn = document.getElementById('toggleShowAllBtn');
        if (!btn) {
            return;
        }
        if (this.showAll) {
            btn.textContent = '❌ Close All';
            btn.style.opacity = '0.7';
        } else {
            btn.textContent = '👁️ Show All';
            btn.style.opacity = '1';
        }
    }

    getActiveGroupLabels() {
        if (this.showAll) {
            return new Set();
        }
        return new Set(this.expandedGroups);
    }
    
    getAllGroupLabels() {
        const labels = new Set();
        // include labels from detected boxes
        this.boxes.forEach(b => {
            labels.add(b.label || 'individual');
        });
        // include coarse groups (even if empty)
        if (Array.isArray(this.coarseGroups)) {
            this.coarseGroups.forEach(g => labels.add(`group_${g.groupId}`));
        }
        return Array.from(labels);
    }
    
    selectGroup(groupLabel) {
        this.selectedGroup = groupLabel;
        // Expand the selected group
        if (groupLabel !== 'individual') {
            this.expandedGroups.add(groupLabel);
        }
        this.selectedBox = null;
        this.updateGroupDisplay();
        this.drawBoxes();
    }
    
    selectPerson(idx, groupLabel) {
        // If selecting a person, show the entire group and highlight the person.
        // Clicking the same person again cancels selection and returns to normal state.
        if (this.selectedBox === idx) {
            // Deselect: simply clear selection (do not modify expandedGroups)
            this.selectedBox = null;
            this.selectedGroup = null;
        } else {
            // Select person: highlight them but do not alter expandedGroups
            this.selectedBox = idx;
            this.selectedGroup = groupLabel || null;
        }
        this.updateGroupDisplay();
        this.drawBoxes();
    }
    
    makeAllIndividual(groupLabel) {
        // Convert all boxes in a group to individual
        let convertedCount = 0;
        this.boxes.forEach(box => {
            if (box.label === groupLabel) {
                box.label = 'individual';
                convertedCount++;
            }
        });
        
        if (convertedCount > 0) {
            console.log(`Converted ${convertedCount} boxes from ${groupLabel} to individual`);
            // Clear selection
            this.selectedBox = null;
            this.selectedGroup = null;
            // Renumber groups to fill gaps
            this.renumberGroups();
            this.updateGroupDisplay();
            this.drawBoxes();
        }
    }
    
    getFirstGroupLabel() {
        for (const box of this.boxes) {
            if (box.label !== 'individual') {
                return box.label;
            }
        }
        // If no groups, return first individual
        if (this.boxes.length > 0) {
            return this.boxes[0].label;
        }
        return null;
    }
    
    highlightDetectionItem(idx) {
        // Keep this method for compatibility but it's now handled by updateGroupDisplay
        // No need to do anything here anymore
    }
    
    deleteBox(idx) {
        const removed = this.boxes[idx];
        if (removed) {
            this.deletedBoxes.push({
                tl_x: removed.tl_x,
                tl_y: removed.tl_y,
                br_x: removed.br_x,
                br_y: removed.br_y,
                label: removed.label,
                deletedAt: new Date().toISOString()
            });
        }
        this.boxes.splice(idx, 1);
        this.selectedBox = null;
        
        // If the selected group has no more boxes, select a different group
        if (this.selectedGroup && !this.boxes.some(b => b.label === this.selectedGroup)) {
            const newGroup = this.getFirstGroupLabel();
            this.selectedGroup = newGroup;
            if (newGroup) {
                this.expandedGroups.add(newGroup);
            }
        }
        
        document.getElementById('boxCount').textContent = this.boxes.length;
        this.updateGroupDisplay();
        this.drawBoxes();
    }
    
    renumberGroups() {
        // Deprecated: preserve backend labels; ensure groupId property exists for each box
        this.boxes.forEach(b => {
            if (b.label && b.label.startsWith('group_')) {
                const m = b.label.match(/(\d+)/);
                b.groupId = m ? parseInt(m[1], 10) : undefined;
            } else {
                b.groupId = undefined;
            }
        });
    }
    
    editLabel(idx) {
        const box = this.boxes[idx];
        if (!box) return;
        
        // Get list of all available groups (from existing boxes AND coarseGroups)
        const availableGroups = new Set();
        this.boxes.forEach(b => {
            if (b.label && b.label.startsWith('group_')) {
                availableGroups.add(b.label);
            }
        });
        // Include coarse groups so empty groups are selectable
        if (Array.isArray(this.coarseGroups)) {
            this.coarseGroups.forEach(g => {
                availableGroups.add(`group_${g.groupId}`);
            });
        }
        
        // Create dialog to select new label
        const groupArray = Array.from(availableGroups).sort();
        let options = groupArray.map(g => `<option value="${g}">${g}</option>`).join('');
        options += `<option value="individual">individual</option>`;
        
        // Add option for new group
        const nextGroupNum = groupArray.length > 0 ? 
            Math.max(...groupArray.map(g => parseInt(g.match(/\d+/)[0]))) + 1 : 1;
        options += `<option value="new_group">+ Create new group (group_${nextGroupNum})</option>`;
        
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        `;
        
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            padding: 30px;
            z-index: 1000;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
            min-width: 350px;
            color: white;
        `;
        
        dialog.innerHTML = `
            <h3 style="margin: 0 0 10px 0; color: white; font-size: 18px;">Change Label</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0 0 20px 0; font-size: 14px;">Current: <strong>${box.label}</strong></p>
            <select id="newLabelSelect" style="width: 100%; padding: 10px; margin-bottom: 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px;">
                ${options}
            </select>
            <div style="display: flex; gap: 10px;">
                <button id="cancelLabel" style="flex: 1; padding: 10px; background: rgba(255,255,255,0.2); border: 2px solid white; border-radius: 6px; cursor: pointer; color: white; font-weight: bold; transition: all 0.2s;">Cancel</button>
                <button id="confirmLabel" style="flex: 1; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.2s;">OK</button>
            </div>
        `;
        
        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        
        const select = document.getElementById('newLabelSelect');
        select.value = box.label;
        select.focus();
        
        // Button handlers
        const closeDialog = () => {
            overlay.remove();
            dialog.remove();
        };
        
        document.getElementById('cancelLabel').onclick = closeDialog;
        
        document.getElementById('confirmLabel').onclick = () => {
            let newLabel = select.value;
            if (newLabel === 'new_group') {
                newLabel = `group_${nextGroupNum}`;
            }
            
            if (newLabel && this.boxes[idx]) {
                this.boxes[idx].label = newLabel;
                // Renumber groups after label change
                this.renumberGroups();
                this.updateGroupDisplay();
                this.drawBoxes();
            }
            closeDialog();
        };
        
        // Keyboard support
        select.onkeydown = (e) => {
            if (e.key === 'Enter') {
                document.getElementById('confirmLabel').click();
            } else if (e.key === 'Escape') {
                closeDialog();
            }
        };
        
        // Close on overlay click
        overlay.onclick = closeDialog;
    }
    
    confirmEditLabel(idx) {
        // Logic moved to editLabel method for better dialog handling
    }
    
    toggleCoarseGroups() {
        this.showCoarseGroups = !this.showCoarseGroups;
        const btn = document.getElementById('toggleCoarseBtn');
        if (this.showCoarseGroups) {
            btn.textContent = '📦 Hide Coarse Groups';
            btn.style.opacity = '0.6';
        } else {
            btn.textContent = '📦 Show Coarse Groups';
            btn.style.opacity = '1';
        }
        this.drawBoxes();
    }
    
    playVideo() {
        if (!this.videoPlayStartTime) {
            this.videoPlayStartTime = Date.now();
        }
        this.logVideoAction('play', { startFrame: this.currentFrame });
        
        this.videoPlaying = true;
        this.videoStartTime = Date.now() - (this.currentFrame * 1000 / 5); // 5 fps, store offset
        this.playBtn.style.display = 'none';
        this.pauseBtn.style.display = 'flex';
        document.getElementById('statusIndicator').className = 'status-indicator status-playing';
        document.getElementById('watchedStatus').textContent = '▶️ Video Playing';
        
        this.videoPlayLoop();
    }
    
    videoPlayLoop() {
        if (!this.videoPlaying) return;
        
        const elapsed = (Date.now() - this.videoStartTime) / 1000 * 5; // 5 fps (slower)
        this.currentFrame = Math.floor(elapsed) % 50;
        
        document.getElementById('frameCounter').textContent = this.currentFrame; // 0-based
        document.getElementById('videoScrubber').value = this.currentFrame;
        
        this.updateVideoFrame();
        
        if (this.videoPlaying) {
            requestAnimationFrame(() => this.videoPlayLoop());
        }
    }
    
    pauseVideo() {
        if (this.videoPlayStartTime) {
            this.totalWatchTime += Date.now() - this.videoPlayStartTime;
            this.videoPlayStartTime = null;
        }
        this.logVideoAction('pause', {
            pausedAtFrame: this.currentFrame,
            totalWatchTimeMs: this.totalWatchTime
        });
        
        this.videoPlaying = false;
        this.pauseBtn.style.display = 'none';
        this.playBtn.style.display = 'flex';
        document.getElementById('statusIndicator').className = 'status-indicator status-paused';
        document.getElementById('watchedStatus').textContent = '⏸️ Video Paused';
    }
    
    updateVideoFrame() {
        const video = this.videos[this.currentVideoIndex];
        const frameNum = this.currentFrame;
        const frameUrl = `/api_finegrained/video/${video.index}/frame/${frameNum}`;
        this.videoImage.src = frameUrl;
    }
    
    onKeyDown(e) {
        if (e.code === 'Space') {
            e.preventDefault();
            if (this.videoPlaying) {
                this.pauseVideo();
            } else {
                this.playVideo();
            }
        } else if (e.code === 'ArrowRight') {
            const prev = this.currentFrame;
            this.currentFrame = (this.currentFrame + 1) % 50;
            this.logVideoAction('frame_next', { fromFrame: prev, toFrame: this.currentFrame, method: 'keyboard' });
            document.getElementById('frameCounter').textContent = this.currentFrame;
            document.getElementById('videoScrubber').value = this.currentFrame;
            this.updateVideoFrame();
        } else if (e.code === 'ArrowLeft') {
            const prev = this.currentFrame;
            this.currentFrame = (this.currentFrame - 1 + 50) % 50;
            this.logVideoAction('frame_previous', { fromFrame: prev, toFrame: this.currentFrame, method: 'keyboard' });
            document.getElementById('frameCounter').textContent = this.currentFrame;
            document.getElementById('videoScrubber').value = this.currentFrame;
            this.updateVideoFrame();
        }
    }
    
    async submitVideo() {
        //this.submitBtn.disabled = true;

        if (this.isPlaying) {
            this.pause();
        }

        try {
            await this.saveAnnotation();
            if (this.currentVideoIndex < this.videos.length - 1) {
                // Prevent the next video's detect() from auto-expanding groups
                this.expandOnDetect = false;
                this.loadVideo(this.currentVideoIndex + 1);
            } else {
                window.location.href = "/finegrained_thank_you";
            }
        } catch (error) {
            console.error('Error saving annotation:', error);
            alert('Error saving annotation. Please check the console.');
        }
    }
    
    async saveAnnotation() {
        try {
            const video = this.videos[this.currentVideoIndex];

            // --- active groups: only boxes that have been assigned a group ---
            const groups = this.boxes.map((box) => ({
                groupId: box.label === 'individual'
                  ? -1
                  : parseInt(box.label.match(/\d+/)?.[0] ?? NaN, 10),
                label: box.label,
                bbox: [
                    Math.round(box.tl_x),
                    Math.round(box.tl_y),
                    Math.round(box.br_x),
                    Math.round(box.br_y)
                ],
                isDeleted: false
            }));

            // --- deleted groups: boxes the annotator removed during this video ---
            const deletedGroups = this.deletedBoxes.map((box, i) => ({
                groupId: `deleted_${i + 1}`,
                label: box.label,
                bbox: [
                    Math.round(box.tl_x),
                    Math.round(box.tl_y),
                    Math.round(box.br_x),
                    Math.round(box.br_y)
                ],
                deletedAt: box.deletedAt,
                isDeleted: true
            }));

            // flush any watch-time that is still accumulating (video still playing)
            const currentWatchTime = this.totalWatchTime +
                (this.videoPlayStartTime ? (Date.now() - this.videoPlayStartTime) : 0);

            const annotationData = {
                // --- identity & location ---
                timestamp: new Date().toISOString(),
                videoIndex: this.currentVideoIndex + 1,
                globalIndex: video.index + 1,
                videoFolder: video.folder,
                annotatedFrame: this.annotatedFrame,

                // --- box payloads ---
                numberOfGroups: groups.length,
                numberOfDeletedGroups: deletedGroups.length,
                groups: groups,
                deletedGroups: deletedGroups,

                // --- video-session metadata (mirrors app.js videoInfo) ---
                videoInfo: {
                    totalFrames: this.totalFrames,
                    annotationFrame: this.annotatedFrame,
                    coordinateSystem: 'normalized',
                    normalizedWidth: this.NORMALIZED_WIDTH,
                    normalizedHeight: this.NORMALIZED_HEIGHT
                },

                // --- watch / timing ---
                totalWatchTimeMs: currentWatchTime,
                videoActionHistory: this.videoActionHistory,
                sessionStartTime: this.sessionStartTime,
                videoLoadTime: this.videoLoadTime,
                annotationDuration: this.videoLoadTime
                    ? Date.now() - new Date(this.videoLoadTime).getTime()
                    : 0
            };

            const res = await fetch('/api_finegrained/save-annotation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(annotationData)
            });

            if (!res.ok) {
                console.error('Save failed');
            }
        } catch (error) {
            console.error('Error saving annotation:', error);
        }
    }
    
    showClearConfirmation() {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        `;

        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            padding: 30px;
            z-index: 1000;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
            min-width: 350px;
            color: white;
            text-align: center;
        `;

        dialog.innerHTML = `
            <h3 style="margin: 0 0 10px 0; color: white; font-size: 18px;">Clear Temporary Files?</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0 0 20px 0; font-size: 14px;">
                The annotation files have been saved to your download folder.<br>
                You can now clear the temporary annotation files to prepare for the next session.
            </p>
            <div style="display: flex; gap: 10px;">
                <button id="keepBtn" style="flex: 1; padding: 10px; background: rgba(255,255,255,0.2); border: 2px solid white; border-radius: 6px; cursor: pointer; color: white; font-weight: bold;">Keep Files</button>
                <button id="clearBtn" style="flex: 1; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">Clear Files</button>
            </div>
        `;

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);

        const closeDialog = () => {
            overlay.remove();
            dialog.remove();
        };

        document.getElementById('keepBtn').onclick = closeDialog;

        document.getElementById('clearBtn').onclick = async () => {
            try {
                await fetch('/api_finegrained/clear-annotations', { method: 'POST' });
                this.showSuccess('✅ Annotation files cleared. Ready for next session!');
            } catch (e) {
                console.log('Clear completed');
            }
            closeDialog();
        };

        overlay.onclick = closeDialog;
    }
    
    showError(message) {
        const elem = document.createElement('div');
        elem.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #fee; color: #c00; border: 1px solid #fcc; padding: 15px; border-radius: 4px; z-index: 10000;';
        elem.textContent = message;
        document.body.appendChild(elem);
        setTimeout(() => elem.remove(), 5000);
    }
    
    showSuccess(message) {
        const elem = document.createElement('div');
        elem.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #efe; color: #0c0; border: 1px solid #cfc; padding: 15px; border-radius: 4px; z-index: 10000;';
        elem.textContent = message;
        document.body.appendChild(elem);
        setTimeout(() => elem.remove(), 5000);
    }
}

// Initialize tool
let tool;
document.addEventListener('DOMContentLoaded', () => {
    tool = new VideoAnnotationTool();
});
