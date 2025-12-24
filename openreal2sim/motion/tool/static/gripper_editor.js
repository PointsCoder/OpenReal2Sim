let scene, camera, renderer, controls, raycaster, mouse;
let sceneGroup, gripperGroup, gripperMesh;
let backgroundGroup, objectsGroup;
let objectMeshes = {}; // Store object meshes by ID
let manipulatedObjectId = null; // Currently manipulated object ID
let sceneLoaded = false;
let gripperSelected = false;
let currentGripperPose = null;
let isDragging = false;
let dragStart = new THREE.Vector2();
let dragStartPose = null;
let rotationSpeed = 0.02;
let translationSpeed = 0.002; // Reduced movement speed
let savedGrasps = []; // List of saved grasps for current object
let graspVisualizations = []; // Three.js objects for visualizing saved grasps
// Removed selectedAxis - no longer needed

// Keyboard state
const keys = {};

function initViewer() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);
    
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(2, 2, 2);
    camera.lookAt(0, 0, 0);
    
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    
    // Grid and axes
    const gridHelper = new THREE.GridHelper(10, 10);
    scene.add(gridHelper);
    const axesHelper = new THREE.AxesHelper(1);
    scene.add(axesHelper);
    
    // Groups - separate background and objects
    sceneGroup = new THREE.Group();
    backgroundGroup = new THREE.Group();
    objectsGroup = new THREE.Group();
    gripperGroup = new THREE.Group();
    gripperGroup.matrixAutoUpdate = false; // Manual matrix control for gripper
    sceneGroup.add(backgroundGroup);
    sceneGroup.add(objectsGroup);
    scene.add(sceneGroup);
    scene.add(gripperGroup);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);
    
    // Event listeners
    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    
    animate();
}

function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    
    // Handle keyboard input
    if (gripperSelected) {
        handleKeyboardInput();
    }
    
    controls.update();
    renderer.render(scene, camera);
}

function onMouseDown(event) {
    // Only handle left mouse button for dragging
    if (event.button !== 0) return;
    
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    
    // Update matrix world for accurate raycasting
    if (gripperGroup) {
        gripperGroup.updateMatrixWorld(true);
    }
    
    // Check if clicking on gripper for dragging
    let intersects = [];
    if (gripperGroup && gripperGroup.children.length > 0) {
        intersects = raycaster.intersectObjects([gripperGroup], true);
        if (intersects.length === 0) {
            intersects = raycaster.intersectObjects(gripperGroup.children, true);
        }
    }
    
    if (intersects.length > 0) {
        // Start dragging
        isDragging = true;
        dragStart.set(event.clientX, event.clientY);
        dragStartPose = gripperGroup.matrix.clone();
        controls.enabled = false;
        event.preventDefault();
    } else {
        // Clicked elsewhere - allow camera control
        isDragging = false;
        controls.enabled = true;
    }
}

function onMouseMove(event) {
    if (isDragging && gripperSelected && dragStartPose) {
        const deltaX = (event.clientX - dragStart.x) * translationSpeed * 0.1;
        const deltaY = -(event.clientY - dragStart.y) * translationSpeed * 0.1;
        
        // Get camera right and up vectors for screen-space translation
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
        camera.getWorldDirection(new THREE.Vector3());
        right.setFromMatrixColumn(camera.matrixWorld, 0);
        up.setFromMatrixColumn(camera.matrixWorld, 1);
        
        const translation = right.multiplyScalar(deltaX).add(up.multiplyScalar(deltaY));
        const currentPos = new THREE.Vector3();
        const currentQuat = new THREE.Quaternion();
        const currentScale = new THREE.Vector3();
        dragStartPose.decompose(currentPos, currentQuat, currentScale);
        
        const newPose = new THREE.Matrix4();
        newPose.compose(currentPos.add(translation), currentQuat, currentScale);
        
        gripperGroup.matrix.copy(newPose);
        currentGripperPose = newPose.clone();
        updatePoseDisplay();
        sendPoseUpdate();
    }
}

function onMouseUp(event) {
    isDragging = false;
    controls.enabled = true;
}

function onKeyDown(event) {
    // Handle arrow keys specially
    let key = event.key.toLowerCase();
    if (event.key.startsWith('Arrow')) {
        key = event.key.toLowerCase().replace('arrow', '');
    }
    keys[key] = true;
}

function onKeyUp(event) {
    // Handle arrow keys specially
    let key = event.key.toLowerCase();
    if (event.key.startsWith('Arrow')) {
        key = event.key.toLowerCase().replace('arrow', '');
    }
    keys[key] = false;
}

function handleKeyboardInput() {
    // Gripper is always selected, just check if pose exists
    if (!currentGripperPose) return;
    
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    currentGripperPose.decompose(pos, quat, scale);
    
    let translation = new THREE.Vector3();
    let rotation = new THREE.Euler();
    
    // Translation - 6 directions: forward/backward, left/right, up/down
    if (keys['w']) translation.z += translationSpeed;  // Forward
    if (keys['s']) translation.z -= translationSpeed;  // Backward
    if (keys['a']) translation.x -= translationSpeed;  // Left
    if (keys['d']) translation.x += translationSpeed;  // Right
    if (keys['q']) translation.y += translationSpeed;  // Up
    if (keys['e']) translation.y -= translationSpeed;  // Down
    
    // Rotation - 3 axes: X, Y, Z (moved to WASD area)
    let rotDelta = new THREE.Euler();
    
    // X axis rotation (R/F keys)
    if (keys['r']) rotDelta.x = rotationSpeed;   // R rotates X positive
    if (keys['f']) rotDelta.x = -rotationSpeed;  // F rotates X negative
    
    // Y axis rotation (T/G keys)
    if (keys['t']) rotDelta.y = -rotationSpeed;  // T rotates Y counter-clockwise
    if (keys['g']) rotDelta.y = rotationSpeed;   // G rotates Y clockwise
    
    // Z axis rotation (Y/H keys)
    if (keys['y']) rotDelta.z = rotationSpeed;   // Y rotates Z positive
    if (keys['h']) rotDelta.z = -rotationSpeed;  // H rotates Z negative
    
    // Apply transformations
    if (translation.length() > 0 || rotDelta.x !== 0 || rotDelta.y !== 0 || rotDelta.z !== 0) {
        // Get current pose
        const currentPos = new THREE.Vector3();
        const currentQuat = new THREE.Quaternion();
        const currentScale = new THREE.Vector3();
        currentGripperPose.decompose(currentPos, currentQuat, currentScale);
        
        // Apply translation
        const newPos = currentPos.clone().add(translation);
        
        // Apply rotation: create quaternion from delta euler and multiply with current quaternion
        // Use XYZ order for Euler angles
        rotDelta.order = 'XYZ';
        const rotQuat = new THREE.Quaternion().setFromEuler(rotDelta);
        // Multiply in local space: new rotation = current rotation * delta rotation
        const newQuat = currentQuat.clone().multiply(rotQuat);
        
        // Compose new pose matrix
        const newPose = new THREE.Matrix4();
        newPose.compose(newPos, newQuat, currentScale);
        
        // Update gripper group matrix (this makes it visible immediately)
        gripperGroup.matrix.copy(newPose);
        gripperGroup.matrixAutoUpdate = false; // We're manually controlling the matrix
        
        // Update current pose
        currentGripperPose = newPose.clone();
        updatePoseDisplay();
        sendPoseUpdate();
    }
}

function updateSelectionUI() {
    const statusEl = document.getElementById('selection-status');
    // Gripper is always selected and green by default
    gripperSelected = true;
    statusEl.textContent = '✓ Gripper Ready (GREEN) - Use keyboard to transform';
    statusEl.className = 'status success';
    
    // Keep gripper green
    if (gripperMesh) {
        gripperMesh.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.color.setHex(0x00ff00); // Always green
                child.material.needsUpdate = true;
                
                if (child.material.emissive !== undefined) {
                    if (!child.material.emissive) {
                        child.material.emissive = new THREE.Color();
                    }
                    child.material.emissive.setHex(0x00ff00);
                    child.material.emissiveIntensity = 0.5;
                }
            }
        });
    }
}

function updateObjectListUI() {
    const objectListEl = document.getElementById('object-list');
    const objectIds = Object.keys(objectMeshes);
    
    if (objectIds.length === 0) {
        objectListEl.innerHTML = '<div style="font-size: 12px; color: #666;">No objects loaded</div>';
        return;
    }
    
    let html = '<div style="font-size: 12px; margin-bottom: 8px; color: #333;"><strong>Objects:</strong></div>';
    
    for (const objId of objectIds) {
        const objData = objectMeshes[objId];
        const objName = objData.name || objId;
        const isManipulated = objId === manipulatedObjectId;
        const color = isManipulated ? '#ff8800' : '#666';
        const bgColor = isManipulated ? '#fff3e0' : 'transparent';
        const border = isManipulated ? '2px solid #ff8800' : '1px solid #ddd';
        
        html += `<div style="padding: 8px; margin: 4px 0; border: ${border}; border-radius: 4px; 
                            background: ${bgColor}; font-size: 12px;">
                    <div style="color: ${color}; font-weight: ${isManipulated ? 'bold' : 'normal'};">
                        ${isManipulated ? '● ' : '○ '}${objName}
                    </div>
                    <div style="font-size: 10px; color: #999; margin-top: 2px;">ID: ${objId}</div>
                    ${isManipulated ? '<div style="color: #ff8800; margin-top: 4px; font-size: 11px;">(ORANGE - Manipulated)</div>' : ''}
                 </div>`;
    }
    
    objectListEl.innerHTML = html;
}

function setManipulatedObject(objId, updateUI = true) {
    // Reset previous manipulated object color
    if (manipulatedObjectId && objectMeshes[manipulatedObjectId]) {
        const prevObjData = objectMeshes[manipulatedObjectId];
        const prevObj = prevObjData.mesh || prevObjData;  // Handle both formats
        prevObj.traverse((child) => {
            if (child.isMesh && child.material) {
                // Reset to original/default color (white/gray)
                child.material.color.setHex(0xffffff);
                child.material.needsUpdate = true;
                if (child.material.emissive !== undefined) {
                    child.material.emissive.setHex(0x000000);
                    child.material.emissiveIntensity = 0;
                }
            }
        });
    }
    
    // Set new manipulated object (only called automatically from scene.json)
    manipulatedObjectId = objId;
    
    // Color the manipulated object orange
    if (objectMeshes[objId]) {
        const objData = objectMeshes[objId];
        const obj = objData.mesh || objData;  // Handle both formats
        const objName = objData.name || objId;
        
        obj.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.color.setHex(0xff8800); // Orange
                child.material.needsUpdate = true;
                if (child.material.emissive !== undefined) {
                    if (!child.material.emissive) {
                        child.material.emissive = new THREE.Color();
                    }
                    child.material.emissive.setHex(0xff8800);
                    child.material.emissiveIntensity = 0.3;
                }
            }
        });
        console.log(`Object ${objName} (${objId}) marked as manipulated (ORANGE)`);
    }
    
    if (updateUI) {
        updateObjectListUI();
    }
}

// Removed updateAxisUI - no longer needed

async function loadScene() {
    const sceneKey = document.getElementById('scene-key').value;
    if (!sceneKey) {
        showStatus('load-status', 'Please enter a scene key', 'error');
        return;
    }
    
    showStatus('load-status', 'Loading scene...', 'success');
    
    try {
        console.log('Loading scene:', sceneKey);
        const response = await fetch('/api/load_scene', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: sceneKey })
        });
        
        console.log('Response status:', response.status, response.statusText);
        const data = await response.json();
        console.log('Response data:', data);
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load scene');
        }
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to load scene');
        }
        
        // Clear existing meshes
        while (backgroundGroup.children.length > 0) {
            backgroundGroup.remove(backgroundGroup.children[0]);
        }
        while (objectsGroup.children.length > 0) {
            objectsGroup.remove(objectsGroup.children[0]);
        }
        while (gripperGroup.children.length > 0) {
            gripperGroup.remove(gripperGroup.children[0]);
        }
        objectMeshes = {}; // Clear object meshes
        manipulatedObjectId = null; // Reset manipulated object (will be set from scene.json)
        
        const loader = new THREE.GLTFLoader();
        
        // Load background separately
        loader.load(data.background, (gltf) => {
            backgroundGroup.add(gltf.scene);
            console.log('Background loaded');
        }, undefined, (error) => {
            console.error('Error loading background:', error);
            showStatus('load-status', 'Error loading background: ' + error.message, 'error');
        });
        
        // Load objects separately
        const objectIds = Object.keys(data.objects);
        const objectInfo = data.object_info || {};  // Object names and types
        const manipulatedOid = data.manipulated_oid;  // Manipulated object ID (could be objId or oid)
        let loadedCount = 0;
        let foundManipulatedObjId = null;  // Store the objId that matches manipulated_oid
        
        for (const [objId, objPath] of Object.entries(data.objects)) {
            loader.load(objPath, (gltf) => {
                const objScene = gltf.scene;
                objScene.name = 'object_' + objId;
                
                // Store reference to object mesh with name
                const objName = objectInfo[objId]?.name || objId;
                const objOid = objectInfo[objId]?.oid || objId;  // Get oid from objectInfo
                objectMeshes[objId] = {
                    mesh: objScene,
                    name: objName,
                    id: objId,
                    oid: objOid,
                };
                
                // Add to objects group
                objectsGroup.add(objScene);
                
                loadedCount++;
                console.log(`Object ${objName} (${objId}, oid: ${objOid}) loaded (${loadedCount}/${objectIds.length})`);
                
                // Check if this is the manipulated object (match by objId or oid)
                if (manipulatedOid) {
                    if (objId === manipulatedOid || objId === String(manipulatedOid) ||
                        objOid === manipulatedOid || objOid === String(manipulatedOid)) {
                        foundManipulatedObjId = objId;
                        console.log(`Found manipulated object: ${objName} (objId: ${objId}, oid: ${objOid}, manipulated_oid: ${manipulatedOid})`);
                    }
                }
                
                // Update object list in UI and set manipulated object after all objects are loaded
                if (loadedCount === objectIds.length) {
                    // Set manipulated object if found
                    if (foundManipulatedObjId) {
                        setManipulatedObject(foundManipulatedObjId, false);  // false = don't update UI yet
                        console.log(`Automatically set manipulated object: ${objectMeshes[foundManipulatedObjId].name} (${foundManipulatedObjId})`);
                    } else if (manipulatedOid) {
                        console.warn(`Manipulated object ID '${manipulatedOid}' not found in loaded objects`);
                    }
                    updateObjectListUI();
                }
            }, undefined, (error) => {
                console.error(`Error loading object ${objId}:`, error);
                loadedCount++;
                // Continue even if one object fails
                if (loadedCount === objectIds.length) {
                    // Set manipulated object if found
                    if (foundManipulatedObjId) {
                        setManipulatedObject(foundManipulatedObjId, false);
                    }
                    updateObjectListUI();
                }
            });
        }
        
        // Load gripper
        loader.load(data.gripper, (gltf) => {
            gripperMesh = gltf.scene;
            
            // Flip gripper mesh upside down (rotate 180 degrees around X axis)
            // This is only applied to mesh vertices for display purposes
            const flipTransform = new THREE.Matrix4();
            flipTransform.makeRotationX(Math.PI); // 180 degrees
            gripperMesh.applyMatrix4(flipTransform);
            
            // Make gripper mesh green by default
            let meshCount = 0;
            gripperMesh.traverse((child) => {
                if (child.isMesh) {
                    meshCount++;
                    child.material = child.material.clone();
                    // Make gripper green by default
                    if (child.material.color) {
                        child.material.color.setHex(0x00ff00); // Green
                    }
                    // Make it slightly emissive for better visibility
                    if (child.material.emissive !== undefined) {
                        child.material.emissive = new THREE.Color(0x00ff00);
                        child.material.emissiveIntensity = 0.5;
                    }
                    // Ensure mesh is visible and raycastable
                    child.visible = true;
                    child.userData.isGripper = true;
                }
            });
            
            gripperGroup.add(gripperMesh);
            gripperGroup.position.set(0, 0, 0);
            
            // Set up for raycasting
            gripperGroup.updateMatrixWorld(true);
            
            console.log(`Gripper loaded with ${meshCount} meshes (flipped upside down, GREEN by default)`);
            console.log('Gripper position:', gripperGroup.position);
            console.log('Gripper matrix:', gripperGroup.matrix);
            
            // Set gripper as selected by default
            gripperSelected = true;
            updateSelectionUI();
        }, undefined, (error) => {
            console.error('Error loading gripper:', error);
        });
        
        // Initialize gripper pose from backend (identity matrix)
        // Note: The mesh is flipped for display, but the pose matrix is not flipped
        // When saving, we'll apply the flip inverse to get the correct pose relative to original mesh
        currentGripperPose = new THREE.Matrix4().fromArray(data.gripper_pose.flat());
        gripperGroup.matrix.copy(currentGripperPose);
        gripperGroup.matrixAutoUpdate = false; // Ensure manual control
        updatePoseDisplay();
        sceneLoaded = true;
        gripperSelected = true; // Always selected
        updateSelectionUI();
        
        // Clear saved grasps when loading new scene
        savedGrasps = [];
        graspVisualizations.forEach(viz => scene.remove(viz));
        graspVisualizations = [];
        updateGraspListUI();
        
        showStatus('load-status', 'Scene loaded successfully!', 'success');
    } catch (error) {
        showStatus('load-status', 'Error: ' + error.message, 'error');
    }
}

async function sendPoseUpdate() {
    if (!currentGripperPose) return;
    
    // Convert from Three.js column-major to standard row-major 4x4 matrix
    const elements = currentGripperPose.elements;
    const matrixArray = [
        [elements[0], elements[4], elements[8], elements[12]],
        [elements[1], elements[5], elements[9], elements[13]],
        [elements[2], elements[6], elements[10], elements[14]],
        [elements[3], elements[7], elements[11], elements[15]]
    ];
    
    try {
        await fetch('/api/update_pose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                transform: matrixArray
            })
        });
    } catch (error) {
        console.error('Failed to update pose:', error);
    }
}

// Removed selectGripper function - gripper is always selected

function resetPose() {
    const identity = new THREE.Matrix4();
    gripperGroup.matrix.copy(identity);
    gripperGroup.matrixAutoUpdate = false;
    currentGripperPose = identity.clone();
    gripperSelected = true; // Always selected
    updateSelectionUI();
    updatePoseDisplay();
    sendPoseUpdate();
}

async function savePose() {
    const sceneKey = document.getElementById('scene-key').value;
    if (!sceneKey) {
        alert('Please enter a scene key first');
        return;
    }
    
    if (!manipulatedObjectId) {
        alert('Please select a manipulated object first');
        return;
    }
    
    if (!currentGripperPose) {
        alert('No gripper pose to save');
        return;
    }
    
    try {
        // Apply flip inverse transform to get pose relative to original (unflipped) mesh
        // The mesh is displayed flipped (upside down), but we need to save relative to the unflipped state
        // where the gripper opens upward
        // FIXME: This is not correct. The gripper is not flipped upside down.
        const flipTransform = new THREE.Matrix4();
        flipTransform.makeRotationX(Math.PI);  // Inverse of the display flip
        const poseToSave = currentGripperPose.clone().multiply(flipTransform);
        
        // Save as 4x4 array in row-major format (standard numpy format)
        // Convert from Three.js column-major to row-major for numpy compatibility
        const elements = poseToSave.elements;
        const matrixArray = [
            [elements[0], elements[4], elements[8], elements[12]],
            [elements[1], elements[5], elements[9], elements[13]],
            [elements[2], elements[6], elements[10], elements[14]],
            [elements[3], elements[7], elements[11], elements[15]]
        ];
        
        const createNew = document.getElementById('create-new-file').checked;
        
        const response = await fetch('/api/save_pose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                key: sceneKey,
                manipulated_oid: manipulatedObjectId,
                transform: matrixArray,
                create_new: createNew
            })
        });
        
        const data = await response.json();
        if (response.ok) {
            const modeText = data.created_new ? 'New file created' : 'Appended to existing file';
            alert(`Grasp saved successfully!\nObject: ${data.object_name} (${data.object_id})\nTotal grasps: ${data.total_grasps}\n${modeText}\nNPZ: ${data.npz_path}`);
            // Reload grasps to update list
            await loadSavedGrasps();
        } else {
            alert('Error: ' + (data.error || 'Unknown error') + (data.traceback ? '\n\n' + data.traceback : ''));
        }
    } catch (error) {
        alert('Error saving pose: ' + error.message);
    }
}

async function loadSavedGrasps() {
    const sceneKey = document.getElementById('scene-key').value;
    if (!sceneKey) {
        alert('Please enter a scene key first');
        return;
    }
    
    if (!manipulatedObjectId) {
        alert('Please select a manipulated object first');
        return;
    }
    
    try {
        const response = await fetch('/api/load_grasps', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                key: sceneKey,
                manipulated_oid: manipulatedObjectId
            })
        });
        
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load grasps');
        }
        
        savedGrasps = data.grasps || [];
        updateGraspListUI();
        visualizeSavedGrasps();
        
        if (savedGrasps.length === 0) {
            showStatus('load-status', 'No saved grasps found', 'success');
        } else {
            showStatus('load-status', `Loaded ${savedGrasps.length} saved grasp(s)`, 'success');
        }
    } catch (error) {
        alert('Error loading grasps: ' + error.message);
    }
}

function updateGraspListUI() {
    const graspListEl = document.getElementById('grasp-list');
    
    if (savedGrasps.length === 0) {
        graspListEl.innerHTML = '<div style="font-size: 12px; color: #666;">No grasps saved yet</div>';
        return;
    }
    
    let html = `<div style="font-size: 12px; margin-bottom: 8px; color: #333;"><strong>${savedGrasps.length} saved grasp(s):</strong></div>`;
    
    savedGrasps.forEach((grasp, idx) => {
        const pos = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        // Convert from row-major 4x4 to column-major 16-element array for Three.js
        const flatPose = [];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                flatPose[j * 4 + i] = grasp.pose[i][j];
            }
        }
        const matrix = new THREE.Matrix4().fromArray(flatPose);
        matrix.decompose(pos, quat, scale);
        const euler = new THREE.Euler().setFromQuaternion(quat);
        
        html += `<div style="padding: 8px; margin: 4px 0; border: 1px solid #ddd; border-radius: 4px; background: white; font-size: 11px;">
                    <div style="font-weight: bold; margin-bottom: 4px;">Grasp #${idx + 1} (Score: ${grasp.score.toFixed(2)})</div>
                    <div style="color: #666; margin-bottom: 4px;">
                        Pos: [${pos.x.toFixed(3)}, ${pos.y.toFixed(3)}, ${pos.z.toFixed(3)}]<br>
                        Rot: [${THREE.MathUtils.radToDeg(euler.x).toFixed(1)}°, ${THREE.MathUtils.radToDeg(euler.y).toFixed(1)}°, ${THREE.MathUtils.radToDeg(euler.z).toFixed(1)}°]
                    </div>
                    <button onclick="loadGraspPose(${idx})" style="padding: 4px 8px; font-size: 10px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; margin-right: 4px;">Load</button>
                    <button onclick="deleteGrasp(${idx})" style="padding: 4px 8px; font-size: 10px; background: #dc3545; color: white; border: none; border-radius: 3px; cursor: pointer;">Delete</button>
                 </div>`;
    });
    
    graspListEl.innerHTML = html;
}

function visualizeSavedGrasps() {
    // Remove existing visualizations
    graspVisualizations.forEach(viz => {
        scene.remove(viz);
    });
    graspVisualizations = [];
    
    // Add new visualizations (semi-transparent grippers)
    savedGrasps.forEach((grasp, idx) => {
        // Convert from row-major 4x4 to column-major 16-element array for Three.js
        const flatPose = [];
        for (let j = 0; j < 4; j++) {  // column
            for (let i = 0; i < 4; i++) {  // row
                flatPose.push(grasp.pose[i][j]);
            }
        }
        const displayMatrix = new THREE.Matrix4().fromArray(flatPose);
        
        // // Apply flip transform to display on the flipped mesh
        // const flipTransform = new THREE.Matrix4();
        // flipTransform.makeRotationX(Math.PI);  // Flip for display
        // const displayMatrix = flipTransform.clone().multiply(matrix);
        
        // Create a simple visualization (wireframe box or helper)
        const helper = new THREE.BoxHelper(new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.15)), 0xffff00);
        helper.matrix.copy(displayMatrix);
        helper.matrixAutoUpdate = false;
        helper.material.opacity = 0.3;
        helper.material.transparent = true;
        scene.add(helper);
        graspVisualizations.push(helper);
    });
}

function loadGraspPose(index) {
    if (index < 0 || index >= savedGrasps.length) {
        alert('Invalid grasp index');
        return;
    }
    
    const grasp = savedGrasps[index];
    // Loaded pose is relative to the unflipped mesh (where gripper opens upward)
    // Convert from row-major 4x4 to column-major 16-element array for Three.js
    const flatPose = [];
    for (let j = 0; j < 4; j++) {  // column
        for (let i = 0; i < 4; i++) {  // row
            flatPose.push(grasp.pose[i][j]);
        }
    }
    const matrix = new THREE.Matrix4().fromArray(flatPose);
    
    // // Apply flip transform to display on the flipped mesh
    const flipTransform = new THREE.Matrix4();
    flipTransform.makeRotationX(-Math.PI);  // Flip for display
    const displayMatrix = matrix.clone().multiply(flipTransform);
    
    gripperGroup.matrix.copy(displayMatrix);
    gripperGroup.matrixAutoUpdate = false;
    currentGripperPose = displayMatrix.clone();
    
    updatePoseDisplay();
    sendPoseUpdate();
    
    showStatus('load-status', `Loaded grasp #${index + 1}`, 'success');
}

async function deleteGrasp(index) {
    if (index < 0 || index >= savedGrasps.length) {
        alert('Invalid grasp index');
        return;
    }
    
    if (!confirm(`Delete grasp #${index + 1}?`)) {
        return;
    }
    
    const sceneKey = document.getElementById('scene-key').value;
    if (!sceneKey || !manipulatedObjectId) {
        alert('Scene and object must be selected');
        return;
    }
    
    try {
        const sceneKey = document.getElementById('scene-key').value;
        const response = await fetch('/api/delete_grasp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                key: sceneKey,
                manipulated_oid: manipulatedObjectId,
                index: index
            })
        });
        
        const data = await response.json();
        if (response.ok) {
            // Reload grasps to update list
            await loadSavedGrasps();
            alert(`Grasp deleted successfully! Remaining grasps: ${data.total_grasps}`);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error deleting grasp: ' + error.message);
    }
}

function updatePoseDisplay() {
    const display = document.getElementById('pose-display');
    if (currentGripperPose) {
        const matrix = currentGripperPose.elements;
        const pos = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        currentGripperPose.decompose(pos, quat, scale);
        const euler = new THREE.Euler().setFromQuaternion(quat);
        
        display.textContent = 'Position: [' + 
            pos.x.toFixed(3) + ', ' + pos.y.toFixed(3) + ', ' + pos.z.toFixed(3) + ']\n' +
            'Rotation: [' + 
            THREE.MathUtils.radToDeg(euler.x).toFixed(1) + '°, ' +
            THREE.MathUtils.radToDeg(euler.y).toFixed(1) + '°, ' +
            THREE.MathUtils.radToDeg(euler.z).toFixed(1) + '°]';
    }
}

function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = 'status ' + (type || '');
    setTimeout(() => {
        element.textContent = '';
        element.className = '';
    }, 3000);
}

window.addEventListener('load', () => {
    initViewer();
    updateSelectionUI();
    updateObjectListUI();
});
