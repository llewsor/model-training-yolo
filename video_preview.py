import os
import cv2
import time
import datetime

def get_video_list(directory="videos"):
    video_extensions = ('.mp4', '.MP4', '.avi', '.mov', '.mkv')
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(video_extensions)])

def load_video(video_index, video_list):
    if 0 <= video_index < len(video_list):
        cap = cv2.VideoCapture(video_list[video_index])
        if cap.isOpened():
            print(f"Loaded video {video_list[video_index]}")
            return cap
    return None

def save_frame(frame, frame_num):
    image_folder = "screenshots"
    os.makedirs(image_folder, exist_ok=True)
    
    # Format the frame number as a 5-digit number
    frame_str = f"{frame_num:05d}"

    # Get current date and time in YYYYMMDD_HHMMSS format
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct new file name format (timestamp + video + frame number)
    file_base_name = f"{timestamp}_frame_{frame_str}"

    image_filename = os.path.join(image_folder, f"{file_base_name}.jpg")

    # Save the cropped image
    cv2.imwrite(image_filename, frame)

def process_video(video_index, video_list):
    cap = load_video(video_index, video_list)
    if not cap:
        return
        
    paused = False  # Variable to track if the video is paused
    frame_index = 0  # Track current frame index
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in video


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {fps} FPS, {width}x{height}")
    
    
    frame_times = []
    
    while True:
        if not paused or (paused):
            if paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
            ret, frame = cap.read()
            
            # Restart video when it ends
            if not ret:
                print("Video finished. Press R to restart. N for next and P for previous video")
                key = cv2.waitKey(0)  # Wait indefinitely for user input
                if key == ord('r'):
                    frame_index = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Restart video
                    continue
                elif key == ord('n'):  # Next video
                    video_index = (video_index + 1) % len(video_list)  # Loop around if at last video
                    if load_video(video_index, video_list):
                        cap.release()
                        cv2.destroyAllWindows()
                        process_video(video_index, video_list)
                elif key == ord('p'):  # Previous video
                    video_index = (video_index - 1) % len(video_list)  # Loop around if at first video
                    if load_video(video_index, video_list):
                        cap.release()
                        cv2.destroyAllWindows()
                        process_video(video_index, video_list)
                else:
                    break  # Exit if any other key is pressed
            
            start = time.perf_counter()        
            
            # Track elapsed time
            end = time.perf_counter()
            elapsed_time = end - start

            # Maintain a rolling average of frame times
            frame_times.append(elapsed_time)
            if len(frame_times) > 10:
                frame_times.pop(0)

            # Ensure we have enough frames to calculate FPS
            if len(frame_times) > 1:
                avg_time_per_frame = sum(frame_times) / len(frame_times)
                fps_ = 1.0 / avg_time_per_frame
            else:
                fps_ = 0  # Avoid division by zero if no frames are processed

            # Overlay navigation instructions on the frame
            overlay_texts = [
                f"Video: {os.path.basename(video_list[video_index])}",
                f"Resolution: {width}x{height}, FPS: {fps_:.2f}/{fps}",
                f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}",
                "Controls:",
                "ESC - Exit  | SPACE - Pause/Resume",
                "R - Restart | S - Save Frame",
                "D - Next Frame | A - Previous Frame (while paused)",
                "N - Next Video | P - Previous Video (while paused)"
            ]
            
            y_offset = height - (len(overlay_texts) * 30) - 10  # Positioning at the bottom

            for text in overlay_texts:
                cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Resize the frame
            resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # Display the resized frame
            cv2.imshow("Frame", resized_frame)
        
        # Handle key events
        key = cv2.waitKeyEx(0 if paused else 1)

        if key == 27:  # ESC key to exit
            break
        elif key == 32:  # Spacebar to pause/resume
            paused = not paused
            frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not paused: 
                frame_times = []  # Reset frame times when pausing/resuming
        elif key == ord("r"): 
            frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Restart video
            paused = False
        elif paused and key == ord("d"): 
            frame_index = min(frame_index + 1, total_frames - 1)
            continue
        elif paused and key == ord("a"): 
            frame_index = max(frame_index - 1, 0)
        elif paused and key == ord('n'):  # Next video
            video_index = (video_index + 1) % len(video_list)  # Loop around if at last video
            if load_video(video_index, video_list):
                cap.release()
                cv2.destroyAllWindows()
                process_video(video_index, video_list)
        elif paused and key == ord('p'):  # Previous video
            video_index = (video_index - 1) % len(video_list)  # Loop around if at first video
            if load_video(video_index, video_list):
                cap.release()
                cv2.destroyAllWindows()
                process_video(video_index, video_list)
        elif paused and key == ord("s"): # Save frame
            save_frame(frame=frame, frame_num=int(frame_index))

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_list = get_video_list("runs/detect/predict2")
    if not video_list:
        print("No videos found.")
        return
    
    video_index = 0
    process_video(video_index, video_list)

if __name__ == "__main__":
    main()
