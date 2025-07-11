from ultralytics import YOLO

model = YOLO('models/best.pt')

# Run inference on a single image
result = model.predict('input_videos/input_video.webm', conf=0.2, save=True)
#result = model.predict('input_videos/blockbusters.mkv', conf=0.2, save=True)

print(result)
for box in result[0].boxes:
    print(box)

    