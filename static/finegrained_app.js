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
        this.selectedBox = null;
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
            this.showError('Failed to load videos');
        }
    }
    
    async loadVideo(index) {
        this.currentVideoIndex = index;
        this.boxes = [];
        this.coarseGroups = [];
        this.selectedBox = null;
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
        document.getElementById('selectionCoords').innerHTML = 'Detecting persons...';
        
        // Load annotation frame
        const img_frameUrl = `/api_finegrained/video/${video.index}/frame/${this.annotatedFrame}`;
        this.annotationImage.src = img_frameUrl;
        
        // Reset video to frame 0
        const frameUrl = `/api_finegrained/video/${video.index}/frame/0`;
        this.videoImage.src = frameUrl;
        document.getElementById('frameCounter').textContent = '0'; // 0-based
        document.getElementById('videoScrubber').value = '0';
        document.getElementById('totalFrames').textContent = '49';
        
        // Load coarse groups
        try {
            const res = await fetch(`/api_finegrained/coarse-groups/${video.index}/frame/${this.annotatedFrame}`);
            const data = await res.json();
            this.coarseGroups = data.groups || [];
        } catch (error) {
            console.error('Error loading coarse groups:', error);
        }
        
        // Auto-run detection as soon as the annotation frame is loaded
        this.annotationImage.onload = () => {
            this.detect();
        };
    }
    
    setupEventListeners() {
        this.submitBtn.addEventListener('click', () => this.submitVideo());
        
        // Toggle coarse groups
        document.getElementById('toggleCoarseBtn').addEventListener('click', () => this.toggleCoarseGroups());
        
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
        
        // Click on annotation canvas to select boxes
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
                // Renumber groups from 1 after detection
                this.renumberGroups();
                document.getElementById('boxCount').textContent = this.boxes.length;
                this.updateBoxList();
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
        
        // Draw detection boxes FIRST (so they're on top for interaction)
        this.boxes.forEach((box, idx) => {
            const x = box.tl_x * scale + offsetX;
            const y = box.tl_y * scale + offsetY;
            const width = (box.br_x - box.tl_x) * scale;
            const height = (box.br_y - box.tl_y) * scale;
            
            const isSelected = this.selectedBox === idx;
            
            // Draw rectangle
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x);
            rect.setAttribute('y', y);
            rect.setAttribute('width', width);
            rect.setAttribute('height', height);
            rect.setAttribute('fill', isSelected ? `${this.getLabelColor(box.label)}40` : 'rgba(255, 107, 107, 0.2)');
            rect.setAttribute('stroke', this.getLabelColor(box.label));
            rect.setAttribute('stroke-width', isSelected ? '5' : '3');
            rect.setAttribute('data-idx', idx);
            rect.setAttribute('pointer-events', 'all');
            rect.style.cursor = 'pointer';
            
            // Add dblclick handler directly to rect
            rect.addEventListener('dblclick', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.selectedBox = idx;
                this.editLabel(idx);
            });
            
            svg.appendChild(rect);
            
            // Draw label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x + 5);
            text.setAttribute('y', y - 5);
            text.setAttribute('fill', this.getLabelColor(box.label));
            text.setAttribute('font-size', isSelected ? '16px' : '14px');
            text.setAttribute('font-weight', '700');
            text.setAttribute('pointer-events', 'all');
            text.textContent = box.label;
            
            // Add dblclick handler to text as well
            text.addEventListener('dblclick', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.selectedBox = idx;
                this.editLabel(idx);
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
                
                corners.forEach(corner => {
                    const handle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    handle.setAttribute('cx', corner.cx);
                    handle.setAttribute('cy', corner.cy);
                    handle.setAttribute('r', handleSize);
                    handle.setAttribute('fill', handleColor);
                    handle.setAttribute('stroke', 'white');
                    handle.setAttribute('stroke-width', '2');
                    handle.setAttribute('data-idx', idx);
                    handle.setAttribute('data-corner', corner.pos);
                    handle.setAttribute('cursor', 'grab');
                    handle.style.cursor = 'grab';
                    
                    handle.addEventListener('mousedown', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        this.startResizeBox(e, idx, corner.pos, scale, offsetX, offsetY);
                    });
                    
                    svg.appendChild(handle);
                });
            }
        });
        
        // Draw coarse groups LAST (so they're in background and don't interfere)
        if (this.showCoarseGroups) {
            this.coarseGroups.forEach((group, idx) => {
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
                text.textContent = `Group ${idx + 1}`;
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
            
            this.drawBoxes();
        };
        
        const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            this.updateBoxList();
        };
        
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    }
    getLabelColor(label) {
        // Vibrant color palette with high contrast for different groups
        const colors = [
            '#FF1744', '#F50057', '#D500F9', '#651FFF', '#2979F3',
            '#0091FF', '#00B0FF', '#00BCD4', '#00ACC1', '#00897B',
            '#00C853', '#76FF03', '#FFEA00', '#FFC400', '#FF9100',
            '#FF3D00', '#FF5252', '#FF6E40', '#FF7043', '#FF8A65'
        ];
        
        if (label === 'individual') {
            return '#999999';
        }
        
        const match = label.match(/group_(\d+)/);
        if (match) {
            const id = parseInt(match[1]);
            return colors[id % colors.length];
        }
        return '#CCCCCC';
    }
    
    onCanvasMouseDown(e) {
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
                this.selectedBox = idx;
                this.drawBoxes(); // Redraw to highlight
                this.highlightDetectionItem(idx); // Highlight in list
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
                this.selectedBox = idx;
                this.editLabel(idx);
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
                this.selectedBox = idx;
                this.editLabel(idx);
                break;
            }
        }
    }
    
    onCanvasMouseMove(e) {
        // Placeholder for dragging
    }
    
    onCanvasMouseUp() {
        // Placeholder
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
                this.updateBoxList();
                this.drawBoxes();
            }
        });
        
        text.addEventListener('click', () => {
            if (this.selectedBox !== null) {
                this.boxes.splice(this.selectedBox, 1);
                this.selectedBox = null;
                document.getElementById('boxCount').textContent = this.boxes.length;
                this.updateBoxList();
                this.drawBoxes();
            }
        });
        
        g.appendChild(btn);
        g.appendChild(text);
        return g;
    }
    
    updateBoxList() {
        const list = document.getElementById('selectionCoords');
        
        if (this.boxes.length === 0) {
            list.innerHTML = 'No detections';
            return;
        }
        
        // Sort boxes by label for better organization
        // Extract group number from label for proper numeric sorting
        const sortedBoxes = this.boxes.map((box, idx) => ({ box, idx }))
            .sort((a, b) => {
                const labelA = a.box.label;
                const labelB = b.box.label;
                
                // Extract group number if it's a group label
                const groupA = labelA.match(/group_(\d+)/);
                const groupB = labelB.match(/group_(\d+)/);
                
                if (groupA && groupB) {
                    return parseInt(groupA[1]) - parseInt(groupB[1]);
                }
                
                // Put individuals at the end
                if (labelA === 'individual' && labelB !== 'individual') return 1;
                if (labelA !== 'individual' && labelB === 'individual') return -1;
                
                return labelA.localeCompare(labelB);
            });
        
        list.innerHTML = sortedBoxes.map(({ box, idx }) => {
            const color = this.getLabelColor(box.label);
            return `
            <div class="box-item" data-box-idx="${idx}" data-original-border="4px solid ${color}" style="border-left: 4px solid ${color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="color: ${color}; font-weight: bold;">${box.label}</span>
                    <div>
                        <button onclick="tool.editLabel(${idx})" style="background: ${color}; color: white; border: none; padding: 2px 6px; border-radius: 2px; cursor: pointer; font-size: 11px; margin-right: 4px;">Edit</button>
                        <button onclick="tool.deleteBox(${idx})" style="background: #ff6b6b; color: white; border: none; padding: 2px 6px; border-radius: 2px; cursor: pointer; font-size: 11px;">×</button>
                    </div>
                </div>
                <span style="color: #999; font-size: 12px;">(${Math.round(box.tl_x)}, ${Math.round(box.tl_y)}, ${Math.round(box.br_x)}, ${Math.round(box.br_y)})</span>
            </div>
        `;
        }).join('');
        
        // Add click handlers to list items
        list.querySelectorAll('.box-item').forEach((item) => {
            item.addEventListener('click', () => {
                const boxIdx = parseInt(item.dataset.boxIdx);
                this.selectedBox = boxIdx;
                this.drawBoxes();
                this.highlightDetectionItem(boxIdx);
            });
        });
    }
    
    highlightDetectionItem(idx) {
        // Remove previous highlight
        const items = document.querySelectorAll('.box-item');
        items.forEach(item => {
            item.style.background = '';
            item.style.borderLeft = item.dataset.originalBorder || '4px solid #ccc';
        });
        
        // Highlight selected item
        if (idx >= 0 && idx < this.boxes.length) {
            const item = document.querySelector(`[data-box-idx="${idx}"]`);
            if (item) {
                const color = this.getLabelColor(this.boxes[idx].label);
                item.style.background = `${color}20`;
                item.style.borderLeft = `4px solid ${color}`;
                item.style.fontWeight = 'bold';
                item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
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
        document.getElementById('boxCount').textContent = this.boxes.length;
        this.updateBoxList();
        this.drawBoxes();
    }
    
    renumberGroups() {
        // Get all unique group numbers currently in use
        const groupNumbers = new Set();
        this.boxes.forEach(b => {
            if (b.label.startsWith('group_')) {
                const match = b.label.match(/group_(\d+)/);
                if (match) {
                    groupNumbers.add(parseInt(match[1]));
                }
            }
        });
        
        if (groupNumbers.size === 0) return; // No groups to renumber
        
        // Create mapping from old numbers to new sequential numbers
        const sortedNumbers = Array.from(groupNumbers).sort((a, b) => a - b);
        const mapping = {};
        sortedNumbers.forEach((oldNum, newNum) => {
            mapping[oldNum] = newNum + 1;
        });
        
        // Update all boxes with new group numbers
        this.boxes.forEach(b => {
            if (b.label.startsWith('group_')) {
                const match = b.label.match(/group_(\d+)/);
                if (match) {
                    const oldNum = parseInt(match[1]);
                    const newNum = mapping[oldNum];
                    b.label = `group_${newNum}`;
                }
            }
        });
    }
    
    editLabel(idx) {
        const box = this.boxes[idx];
        if (!box) return;
        
        // Get list of all available groups and option to create new group
        const availableGroups = new Set();
        this.boxes.forEach(b => {
            if (b.label.startsWith('group_')) {
                availableGroups.add(b.label);
            }
        });
        
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
                this.updateBoxList();
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
