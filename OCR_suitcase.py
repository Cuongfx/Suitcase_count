import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
from shapely.geometry import Point, Polygon

class RegionPersistentCounter:
    def __init__(self, video_path, region1_points, region2_points,
                 model_path="yolo11n.pt", output_path="region_counting.mp4"):
        self.video_path = video_path
        
        # Create polygons for both regions
        self.region_polygon1 = Polygon(region1_points)
        self.region_polygon2 = Polygon(region2_points)
        
        self.model_path = model_path
        self.output_path = output_path
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened(), f"Error reading video file: {video_path}"
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        # Initialize the model
        try:
            self.model = YOLO(model_path)
            print(f"Model {model_path} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # YOLO track ID -> custom unique ID
        self.yolo_id_to_custom_id = {}
        self.next_custom_id = 1
        
        # Track custom ID -> assigned region: 1 or 2
        self.object_region_assignments = {}
        
        self.colors = Colors()  # For bounding box colors
        
        # Debug variable: class occurrence counts
        self.class_counts = {}

    def is_in_region(self, point_xy, region_polygon):
        """Check if a (x, y) point is inside the given region polygon."""
        point = Point(point_xy)
        return region_polygon.contains(point)
    
    def draw_regions(self, frame):
        """
        Draw Region 1 in light gray (192,192,192)
        and Region 2 in very light blue (255,248,240).
        """
        # Convert shapely polygons to NumPy coordinates
        pts1 = np.array(list(self.region_polygon1.exterior.coords), np.int32).reshape((-1, 1, 2))
        pts2 = np.array(list(self.region_polygon2.exterior.coords), np.int32).reshape((-1, 1, 2))

        overlay = frame.copy()
        
        # Region 1: Light Gray
        cv2.polylines(frame, [pts1], True, (192, 192, 192), 2)
        cv2.fillPoly(overlay, [pts1], (192, 192, 192))
        
        # Region 2: Very Light Blue
        cv2.polylines(frame, [pts2], True, (255, 248, 240), 2)
        cv2.fillPoly(overlay, [pts2], (255, 248, 240))

        # Blend the overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame

    def draw_statistics(self, frame):
        """Overlay statistics text on the frame."""
        region1_count = sum(1 for r in self.object_region_assignments.values() if r == 1)
        region2_count = sum(1 for r in self.object_region_assignments.values() if r == 2)
        total_count = region1_count + region2_count
        
        # Draw total
        cv2.putText(
            frame, f"Total Luggage: {total_count}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )
        
        # Region 1 count
        cv2.putText(
            frame, f"Upper path: {region1_count}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        # Region 2 count
        cv2.putText(
            frame, f"Lower path: {region2_count}",
            (20, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )
        
        return frame

    def process_frame(self, frame, frame_count):
        output_frame = frame.copy()
        
        # Run detection + tracking with YOLO
        results = self.model.track(frame, persist=True, verbose=False)
        
        # Draw both regions
        output_frame = self.draw_regions(output_frame)
        
        # Process detections
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Update class occurrence counts (optional debug)
            if hasattr(results[0].boxes, 'cls') and results[0].boxes.cls is not None:
                classes_this_frame = results[0].boxes.cls.cpu().numpy().astype(int)
                for c in classes_this_frame:
                    self.class_counts[c] = self.class_counts.get(c, 0) + 1
                
                # Print debug info for first few frames
                if frame_count <= 10:
                    unique_classes = np.unique(classes_this_frame)
                    print(f"Frame {frame_count} - Detected classes: {unique_classes}")
            
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                try:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, cls_val, yolo_id in zip(boxes, classes, track_ids):
                        # Filter for suitcase (COCO index 28)
                        if cls_val == 28:
                            x1, y1, x2, y2 = box
                            
                            # Map YOLO's track ID to our custom ID
                            if yolo_id not in self.yolo_id_to_custom_id:
                                self.yolo_id_to_custom_id[yolo_id] = self.next_custom_id
                                self.next_custom_id += 1
                            unique_id = self.yolo_id_to_custom_id[yolo_id]
                            
                            # If object already assigned, skip re-check
                            if unique_id in self.object_region_assignments:
                                assigned_region = self.object_region_assignments[unique_id]
                                color = self.colors(int(unique_id), True)
                                
                                # Draw bounding box + ID
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(
                                    output_frame,
                                    f"ID:{unique_id} (cls={cls_val})",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2
                                )
                                # Draw assigned region label
                                region_label = "R1" if assigned_region == 1 else "R2"
                                cv2.putText(
                                    output_frame,
                                    region_label,
                                    (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2
                                )
                                continue  # don't re-check region
                            
                            # Object not assigned => check corners + midpoints
                            check_points = [
                                (x2, y1),                        # top-right corner
                                (x1, y2),                        # bottom-left corner
                                (x2, y2),                        # bottom-right corner
                                ((x1 + x2) / 2, y2),             # bottom midpoint
                                (x1, (y1 + y2) / 2),             # left midpoint
                                (x2, (y1 + y2) / 2),             # right midpoint
                            ]
                            
                            in_r1 = False
                            in_r2 = False
                            for cp in check_points:
                                if self.is_in_region(cp, self.region_polygon1):
                                    in_r1 = True
                                    break
                                elif self.is_in_region(cp, self.region_polygon2):
                                    in_r2 = True
                                    break
                            
                            # Assign region if inside
                            if in_r1:
                                self.object_region_assignments[unique_id] = 1
                                print(f"Unique ID {unique_id} assigned to Region 1.")
                            elif in_r2:
                                self.object_region_assignments[unique_id] = 2
                                print(f"Unique ID {unique_id} assigned to Region 2.")
                            else:
                                pass  # OUT / unassigned
                            
                            # Draw bounding box
                            color = self.colors(int(unique_id), True)
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                output_frame,
                                f"ID:{unique_id} (cls={cls_val})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                            
                            for cp in check_points:
                                cv2.circle(output_frame, (int(cp[0]), int(cp[1])), 5, color, -1)
                            
                            # Draw assigned region label
                            assigned_region = self.object_region_assignments.get(unique_id, None)
                            if assigned_region == 1:
                                region_label = "R1"
                            elif assigned_region == 2:
                                region_label = "R2"
                            else:
                                region_label = "OUT"
                            
                            cv2.putText(
                                output_frame,
                                region_label,
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                except Exception as e:
                    print(f"Error processing tracking results: {e}")
            elif frame_count % 10 == 0:
                print("No tracking IDs available in this frame.")
        elif frame_count % 10 == 0:
            print("No detections in this frame.")
        
        # Draw overlay statistics
        output_frame = self.draw_statistics(output_frame)
        return output_frame

    def process_video(self):
        """Process the entire video frame by frame."""
        frame_count = 0
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}")
            
            output_frame = self.process_frame(frame, frame_count)
            
            # Display in a window and save to output
            cv2.imshow("Region Persistent Counter", output_frame)
            self.video_writer.write(output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Final counts
        region1_count = sum(1 for r in self.object_region_assignments.values() if r == 1)
        region2_count = sum(1 for r in self.object_region_assignments.values() if r == 2)
        total_count = region1_count + region2_count
        
        print(f"\nProcessing complete. Output saved to {self.output_path}")
        print(f"Total unique suitcases: {total_count}")
        print(f"Upper path count: {region1_count}")
        print(f"Lower path count: {region2_count}")
        
        if self.class_counts:
            print("Class detection statistics:")
            for cls_id, count in self.class_counts.items():
                print(f"  Class {cls_id}: {count} detections")


if __name__ == "__main__":
    # Path to your video file
    video_path = "lugguage_airport.mp4"

    # Define your region polygons here
    region1_points = [(1074, 614), (1714, 485), (1714, 536), (1074, 654)]
    region2_points = [(998, 480), (1025, 1080), (1003, 1080), (982, 480)]
    # region1_points = [(1133, 503), (964, 292), (1036, 229), (1561, 199),(1913, 117),(1916, 190),(1753, 226),(1308, 250),(1624, 434),(1726, 470),(1133, 590)]
    # region2_points = [(1919, 744), (1919, 1063), (0, 1080), (0, 810)]
    # # Output video file
    output_path = "baggage_counting_fast.mp4"
    
    # Initialize and run
    counter = RegionPersistentCounter(
        video_path=video_path,
        region1_points=region1_points,
        region2_points=region2_points,
        model_path="yolo12m.pt",
        output_path=output_path
    )
    counter.process_video()
