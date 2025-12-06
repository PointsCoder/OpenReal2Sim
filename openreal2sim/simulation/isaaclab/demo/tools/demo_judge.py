"""
Demo Judge Tool: Review and filter trajectories (both reference and generated) by watching videos.

For each key, load videos from outputs/<key>/videos/ and display them in Gradio.
Users can accept or reject each video. When rejected:
- Remove corresponding trajectory from task.json (reference_trajectory or generated_trajectories)
- Delete corresponding HDF5 file from h5py/<key>/
- Delete corresponding video file from outputs/<key>/videos/
"""

import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import h5py


BASE_DIR = Path.cwd()
OUTPUTS_DIR = BASE_DIR / "outputs"
TASKS_DIR = BASE_DIR / "tasks"
H5PY_DIR = BASE_DIR / "h5py"
VIDEOS_SUBDIR = "videos"


def get_all_keys() -> List[str]:
    """Get all keys from outputs directory."""
    if not OUTPUTS_DIR.exists():
        return []
    keys = []
    for item in OUTPUTS_DIR.iterdir():
        if item.is_dir():
            videos_dir = item / VIDEOS_SUBDIR
            if videos_dir.exists() and any(videos_dir.glob("*.mp4")):
                keys.append(item.name)
    return sorted(keys)


def get_videos_for_key(key: str) -> List[Tuple[str, Path]]:
    """Get all video files for a given key.
    
    Returns:
        List of tuples: (video_name, video_path)
    """
    videos_dir = OUTPUTS_DIR / key / VIDEOS_SUBDIR
    if not videos_dir.exists():
        return []
    
    videos = []
    for video_file in sorted(videos_dir.glob("*.mp4")):
        videos.append((video_file.name, video_file))
    return videos


def get_trajectory_index_from_video_name(video_name: str) -> Optional[int]:
    """Extract trajectory index from video name.
    
    Video names are like: demo_000.mp4, demo_001.mp4, etc.
    Returns the numeric index (0, 1, 2, ...) or None if parsing fails.
    """
    try:
        # Extract number from demo_XXX.mp4
        name_without_ext = video_name.replace(".mp4", "")
        if name_without_ext.startswith("demo_"):
            index_str = name_without_ext.replace("demo_", "")
            return int(index_str)
    except (ValueError, AttributeError):
        pass
    return None


def load_task_json(key: str) -> Optional[Dict]:
    """Load task.json for a given key."""
    task_json_path = TASKS_DIR / key / "task.json"
    if not task_json_path.exists():
        return None
    with open(task_json_path, "r") as f:
        return json.load(f)


def save_task_json(key: str, task_data: Dict):
    """Save task.json for a given key."""
    task_json_path = TASKS_DIR / key / "task.json"
    task_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(task_json_path, "w") as f:
        json.dump(task_data, f, indent=2)


def detect_trajectory_type(key: str, traj_idx: int) -> Tuple[bool, int]:
    """Detect if trajectory is reference or generated, and return adjusted index.
    
    Returns:
        (is_reference, adjusted_idx): 
        - is_reference: True if in reference_trajectory, False if in generated_trajectories
        - adjusted_idx: Index within the respective list
    """
    task_data = load_task_json(key)
    if task_data is None:
        return True, traj_idx  # Default to reference
    
    ref_trajs = task_data.get("reference_trajectory", [])
    if isinstance(ref_trajs, list) and 0 <= traj_idx < len(ref_trajs):
        return True, traj_idx
    
    # If not in reference, check generated
    gen_trajs = task_data.get("generated_trajectories", [])
    if isinstance(gen_trajs, list):
        # Adjust index: subtract reference trajectory count
        adjusted_idx = traj_idx - len(ref_trajs)
        if 0 <= adjusted_idx < len(gen_trajs):
            return False, adjusted_idx
    
    # Default: assume reference
    return True, traj_idx


def remove_trajectory_from_json(key: str, traj_idx: int, is_reference: bool):
    """Remove trajectory at index from task.json.
    
    Args:
        key: Task key
        traj_idx: Trajectory index to remove (adjusted index within the list)
        is_reference: True if reference_trajectory, False if generated_trajectories
    """
    task_data = load_task_json(key)
    if task_data is None:
        print(f"[WARN] Task JSON not found for key: {key}")
        return False
    
    traj_list_key = "reference_trajectory" if is_reference else "generated_trajectories"
    
    if traj_list_key not in task_data:
        task_data[traj_list_key] = []
    
    traj_list = task_data[traj_list_key]
    if isinstance(traj_list, list) and 0 <= traj_idx < len(traj_list):
        removed = traj_list.pop(traj_idx)
        print(f"[INFO] Removed trajectory {traj_idx} from {traj_list_key}")
        save_task_json(key, task_data)
        return True
    else:
        print(f"[WARN] Trajectory index {traj_idx} out of range for {traj_list_key} (len={len(traj_list)})")
        return False


def delete_hdf5_file(key: str, video_name: str):
    """Delete corresponding HDF5 file for a video.
    
    Video: demo_XXX.mp4 -> HDF5: episode_XXX.hdf5
    """
    traj_idx = get_trajectory_index_from_video_name(video_name)
    if traj_idx is None:
        print(f"[WARN] Could not parse trajectory index from {video_name}")
        return False
    
    hdf5_path = H5PY_DIR / key / f"episode_{traj_idx:03d}.hdf5"
    if hdf5_path.exists():
        hdf5_path.unlink()
        print(f"[INFO] Deleted HDF5: {hdf5_path}")
        return True
    else:
        print(f"[WARN] HDF5 file not found: {hdf5_path}")
        return False


def delete_video_file(key: str, video_name: str):
    """Delete video file.
    
    Args:
        key: Task key
        video_name: Video file name (e.g., demo_000.mp4)
    """
    video_path = OUTPUTS_DIR / key / VIDEOS_SUBDIR / video_name
    if video_path.exists():
        video_path.unlink()
        print(f"[INFO] Deleted video: {video_path}")
        return True
    else:
        print(f"[WARN] Video file not found: {video_path}")
        return False


def process_reject(key: str, video_name: str):
    """Process rejection: delete trajectory, HDF5, and video.
    
    Args:
        key: Task key
        video_name: Video file name (e.g., demo_000.mp4)
    
    Returns:
        (success, message): Tuple of (bool, str)
    """
    print(f"[INFO] Processing rejection for key={key}, video={video_name}")
    
    # Get trajectory index
    traj_idx = get_trajectory_index_from_video_name(video_name)
    if traj_idx is None:
        return False, f"Error: Could not parse trajectory index from {video_name}"
    
    # Detect trajectory type
    is_reference, adjusted_idx = detect_trajectory_type(key, traj_idx)
    traj_type_str = "reference" if is_reference else "generated"
    
    # Remove from JSON
    success = remove_trajectory_from_json(key, adjusted_idx, is_reference=is_reference)
    if not success:
        # Still try to delete files even if JSON removal failed
        print(f"[WARN] Could not remove {traj_type_str} trajectory {adjusted_idx} from JSON, but will still delete files")
    
    # Delete HDF5
    hdf5_deleted = delete_hdf5_file(key, video_name)
    
    # Delete video
    video_deleted = delete_video_file(key, video_name)
    
    result_parts = [f"Rejected {traj_type_str} trajectory {adjusted_idx} for key '{key}'"]
    if hdf5_deleted:
        result_parts.append("HDF5 deleted")
    if video_deleted:
        result_parts.append("Video deleted")
    
    return True, " | ".join(result_parts)


def batch_process_reject(key: str, video_names: List[str]) -> str:
    """Batch process rejections: delete trajectories, HDF5 files, and videos.
    
    Args:
        key: Task key
        video_names: List of video file names to reject
    
    Returns:
        Summary message
    """
    if not video_names:
        return "No videos to reject"
    
    print(f"[INFO] Batch processing {len(video_names)} rejections for key={key}")
    
    results = []
    for video_name in video_names:
        success, message = process_reject(key, video_name)
        if success:
            results.append(f"✓ {video_name}")
        else:
            results.append(f"✗ {video_name}: {message}")
    
    summary = f"Processed {len(video_names)} rejections:\n" + "\n".join(results)
    return summary


def create_gradio_interface():
    """Create Gradio interface for reviewing videos."""
    
    # State to track current key and video index
    current_state = {"key": None, "video_idx": 0, "videos": [], "rejected_videos": []}
    
    def load_key(key: str):
        """Load videos for a key."""
        if not key:
            return None, "Please select a key", gr.update(visible=False), gr.update(visible=False), gr.update(value=1.0), gr.update(value="")
        
        videos = get_videos_for_key(key)
        if not videos:
            return None, f"No videos found for key '{key}'", gr.update(visible=False), gr.update(visible=False), gr.update(value=1.0), gr.update(value="")
        
        current_state["key"] = key
        current_state["videos"] = videos
        current_state["video_idx"] = 0
        current_state["rejected_videos"] = []
        
        video_name, video_path = videos[0]
        video_info = f"Key: {key}\nVideo: {video_name}\n({len(videos)} total videos)"
        rejected_list = ""
        
        return str(video_path), video_info, gr.update(visible=True), gr.update(visible=True), gr.update(value=1.0), gr.update(value=rejected_list)
    
    def next_video(playback_speed: float):
        """Move to next video."""
        if not current_state["videos"]:
            rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
            return gr.update(value=None), "No videos loaded", gr.update(visible=False), gr.update(visible=False), gr.update(value=1.0), gr.update(value=rejected_list)
        
        current_state["video_idx"] = (current_state["video_idx"] + 1) % len(current_state["videos"])
        video_name, video_path = current_state["videos"][current_state["video_idx"]]
        key = current_state["key"]
        
        video_info = f"Key: {key}\nVideo: {video_name}\n({len(current_state['videos'])} total videos, {current_state['video_idx'] + 1}/{len(current_state['videos'])})"
        rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
        
        return str(video_path), video_info, gr.update(visible=True), gr.update(visible=True), gr.update(value=playback_speed), gr.update(value=rejected_list)
    
    def prev_video(playback_speed: float):
        """Move to previous video."""
        if not current_state["videos"]:
            rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
            return gr.update(value=None), "No videos loaded", gr.update(visible=False), gr.update(visible=False), gr.update(value=1.0), gr.update(value=rejected_list)
        
        current_state["video_idx"] = (current_state["video_idx"] - 1) % len(current_state["videos"])
        video_name, video_path = current_state["videos"][current_state["video_idx"]]
        key = current_state["key"]
        
        video_info = f"Key: {key}\nVideo: {video_name}\n({len(current_state['videos'])} total videos, {current_state['video_idx'] + 1}/{len(current_state['videos'])})"
        rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
        
        return str(video_path), video_info, gr.update(visible=True), gr.update(visible=True), gr.update(value=playback_speed), gr.update(value=rejected_list)
   
    
    def accept_video(playback_speed: float):
        """Accept current video: remove from rejected list if it was marked for deletion."""
        if not current_state["videos"]:
            rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
            return "No video to accept", gr.update(value=None), "No videos", gr.update(value=1.0), gr.update(value=rejected_list)
        
        video_name, video_path = current_state["videos"][current_state["video_idx"]]
        key = current_state["key"]
        
        # Remove from rejected list if it's there
        if video_name in current_state["rejected_videos"]:
            current_state["rejected_videos"].remove(video_name)
            message = f"Removed {video_name} from deletion list"
        else:
            message = f"{video_name} is already accepted"
        
        video_info = f"Key: {key}\nVideo: {video_name}\n({len(current_state['videos'])} total videos, {current_state['video_idx'] + 1}/{len(current_state['videos'])})"
        rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
        
        return message, str(video_path), video_info, gr.update(value=playback_speed), gr.update(value=rejected_list)
    
    def reject_video(playback_speed: float):
        """Mark current video as rejected (will be deleted when Save Results is clicked)."""
        if not current_state["videos"]:
            rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
            return "No video to reject", gr.update(value=None), "No videos", gr.update(value=1.0), gr.update(value=rejected_list)
        
        video_name, video_path = current_state["videos"][current_state["video_idx"]]
        key = current_state["key"]
        
        # Add to rejected list if not already there
        if video_name not in current_state["rejected_videos"]:
            current_state["rejected_videos"].append(video_name)
        
        # Move to next video
        current_state["video_idx"] = (current_state["video_idx"] + 1) % len(current_state["videos"])
        
        if current_state["videos"]:
            next_video_name, next_video_path = current_state["videos"][current_state["video_idx"]]
            video_info = f"Key: {key}\nVideo: {next_video_name}\n({len(current_state['videos'])} total videos, {current_state['video_idx'] + 1}/{len(current_state['videos'])})"
            rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
            
            return f"Marked {video_name} for deletion", str(next_video_path), video_info, gr.update(value=playback_speed), gr.update(value=rejected_list)
        else:
            rejected_list = "\n".join(current_state["rejected_videos"]) if current_state["rejected_videos"] else ""
            return f"Marked {video_name} for deletion", None, f"Key: {key}\nNo more videos", gr.update(value=playback_speed), gr.update(value=rejected_list)
    
    def update_playback_speed(playback_speed: float):
        """Update video playback speed - returns the same video to trigger update."""
        if not current_state["videos"]:
            return gr.update()
        
        video_name, video_path = current_state["videos"][current_state["video_idx"]]
        # Return the same video path to trigger reload, which will apply the new speed
        return str(video_path)
    
    def save_results(playback_speed: float):
        """Save results: batch delete all rejected videos and move to next key."""
        if not current_state["rejected_videos"]:
            return "No videos marked for deletion", gr.update(), gr.update(value=""), gr.update()
        
        key = current_state["key"]
        rejected_count = len(current_state["rejected_videos"])
        
        # Batch process rejections
        summary = batch_process_reject(key, current_state["rejected_videos"])
        
        # Get next key
        all_keys = get_all_keys()
        current_key_idx = all_keys.index(key) if key in all_keys else -1
        next_key_idx = current_key_idx + 1
        
        if next_key_idx < len(all_keys):
            next_key = all_keys[next_key_idx]
            # Load next key
            videos = get_videos_for_key(next_key)
            if videos:
                current_state["key"] = next_key
                current_state["videos"] = videos
                current_state["video_idx"] = 0
                current_state["rejected_videos"] = []
                
                video_name, video_path = videos[0]
                video_info = f"Key: {next_key}\nVideo: {video_name}\n({len(videos)} total videos)"
                rejected_list = ""
                
                result_msg = f"Deleted {rejected_count} videos from '{key}'\n\n{summary}\n\nSwitched to next key: {next_key}"
                return result_msg, gr.update(choices=all_keys, value=next_key), str(video_path), video_info, gr.update(visible=True), gr.update(visible=True), gr.update(value=1.0), gr.update(value=rejected_list)
            else:
                result_msg = f"Deleted {rejected_count} videos from '{key}'\n\n{summary}\n\nNo more keys with videos"
                return result_msg, gr.update(choices=all_keys), None, f"Key: {key}\nNo more videos", gr.update(visible=False), gr.update(visible=False), gr.update(value=1.0), gr.update(value="")
        else:
            result_msg = f"Deleted {rejected_count} videos from '{key}'\n\n{summary}\n\nNo more keys"
            return result_msg, gr.update(choices=all_keys), None, f"Key: {key}\nNo more keys", gr.update(visible=False), gr.update(visible=False), gr.update(value=1.0), gr.update(value="")
    
    # Create Gradio interface
    with gr.Blocks(title="Demo Judge") as demo:
        gr.Markdown("# Demo Judge Tool")
        gr.Markdown("Review and filter trajectories (both reference and generated) by watching videos.")
        gr.Markdown("**Accept**: Keep the video (remove from deletion list if marked)")
        gr.Markdown("**Reject**: Mark video for deletion (will be deleted when 'Save Results' is clicked)")
        gr.Markdown("**Save Results**: Delete all marked videos and move to next key")
        
        with gr.Row():
            key_dropdown = gr.Dropdown(
                choices=get_all_keys(),
                label="Select Key",
                interactive=True
            )
            load_btn = gr.Button("Load Key", variant="primary")
        
        with gr.Row():
            video_player = gr.Video(label="Video Player", autoplay=True)
            with gr.Column():
                video_info = gr.Textbox(label="Video Info", interactive=False)
                rejected_list = gr.Textbox(
                    label="Videos Marked for Deletion",
                    interactive=False,
                    lines=5,
                    placeholder="No videos marked for deletion yet"
                )
        
        with gr.Row():
            prev_btn = gr.Button("Previous Video")
            next_btn = gr.Button("Next Video")
            accept_btn = gr.Button("Accept", variant="stop")
            reject_btn = gr.Button("Reject", variant="stop")
        
        with gr.Row():
            save_results_btn = gr.Button("Save Results", variant="primary")
        
        with gr.Row():
            playback_speed_slider = gr.Slider(
                minimum=0.25,
                maximum=3.0,
                value=1.0,
                step=0.25,
                label="Playback Speed",
                info="Adjust video playback speed (0.25x to 3.0x). Use Apply Speed button to update."
            )
            apply_speed_btn = gr.Button("Apply Speed", variant="secondary")
        
        # Add custom JavaScript for autoplay and playback speed control
        demo.load(
            fn=None,
            js="""
            function() {
                // Function to set playback speed and autoplay
                function setupVideo() {
                    const videoElements = document.querySelectorAll('video');
                    videoElements.forEach(video => {
                        if (video) {
                            // Get speed from slider
                            const speedSlider = document.querySelector('input[type="range"]');
                            if (speedSlider) {
                                video.playbackRate = parseFloat(speedSlider.value) || 1.0;
                            }
                            // Autoplay
                            video.play().catch(e => console.log('Autoplay prevented:', e));
                        }
                    });
                }
                
                // Setup video when page loads
                setTimeout(setupVideo, 500);
                
                // Setup video when video element changes
                const observer = new MutationObserver(setupVideo);
                observer.observe(document.body, { childList: true, subtree: true });
            }
            """
        )
        
        result_text = gr.Textbox(label="Result", interactive=False)
        
        # JavaScript function to apply autoplay and playback speed
        def apply_video_settings():
            return None
        
        video_js = """
        function() {
            setTimeout(function() {
                const videoElements = document.querySelectorAll('video');
                const speedSlider = document.querySelector('input[type="range"]');
                const speed = speedSlider ? parseFloat(speedSlider.value) : 1.0;
                videoElements.forEach(video => {
                    if (video) {
                        video.playbackRate = speed;
                        video.play().catch(e => console.log('Autoplay prevented:', e));
                    }
                });
            }, 300);
        }
        """
        
        # Event handlers
        load_btn.click(
            fn=load_key,
            inputs=[key_dropdown],
            outputs=[video_player, video_info, accept_btn, reject_btn, playback_speed_slider, rejected_list]
        ).then(fn=apply_video_settings, js=video_js)
        
        next_btn.click(
            fn=next_video,
            inputs=[playback_speed_slider],
            outputs=[video_player, video_info, accept_btn, reject_btn, playback_speed_slider, rejected_list]
        ).then(fn=apply_video_settings, js=video_js)
        
        prev_btn.click(
            fn=prev_video,
            inputs=[playback_speed_slider],
            outputs=[video_player, video_info, accept_btn, reject_btn, playback_speed_slider, rejected_list]
        ).then(fn=apply_video_settings, js=video_js)
        
        accept_btn.click(
            fn=accept_video,
            inputs=[playback_speed_slider],
            outputs=[result_text, video_player, video_info, playback_speed_slider, rejected_list]
        ).then(fn=apply_video_settings, js=video_js)
        
        reject_btn.click(
            fn=reject_video,
            inputs=[playback_speed_slider],
            outputs=[result_text, video_player, video_info, playback_speed_slider, rejected_list]
        ).then(fn=apply_video_settings, js=video_js)
        
        save_results_btn.click(
            fn=save_results,
            inputs=[playback_speed_slider],
            outputs=[result_text, key_dropdown, video_player, video_info, accept_btn, reject_btn, playback_speed_slider, rejected_list]
        ).then(fn=apply_video_settings, js=video_js)
        
        def apply_speed(playback_speed: float):
            """Apply playback speed to current video."""
            if not current_state["videos"]:
                return gr.update()
            
            video_name, video_path = current_state["videos"][current_state["video_idx"]]
            # Return the same video path to trigger reload, which will apply the new speed via JS
            return str(video_path)
        
        apply_speed_btn.click(
            fn=apply_speed,
            inputs=[playback_speed_slider],
            outputs=[video_player]
        ).then(
            fn=None,
            js="""
            function() {
                setTimeout(function() {
                    const videoElements = document.querySelectorAll('video');
                    const speedSlider = document.querySelector('input[type="range"]');
                    const speed = speedSlider ? parseFloat(speedSlider.value) : 1.0;
                    videoElements.forEach(video => {
                        if (video) {
                            video.playbackRate = speed;
                            video.play().catch(e => console.log('Autoplay prevented:', e));
                        }
                    });
                }, 200);
            }
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7861)

