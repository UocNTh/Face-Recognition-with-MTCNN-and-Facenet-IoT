# Face Recognition with MTCNN and Facenet

## Phần 1: Tìm hiểu khái niệm 

Bài toán chia ra thành 2 vấn đề chính:
- Phát hiện khuôn mặt (Face Detection)
- Phân biệt khuôn mặt (Face Verification)

### 1. Tổng quan về MTCNN (Multi-task Cascaded Convolutional Networks)

Phần đầu tiên sẽ về Face Detection, bài toán đặt ra là phát hiện các khuôn mặt có trong ảnh hoặc trong frame của video. Mạng MTCNN với 3 lớp khác biệt, tượng trưng cho 3 stage chính là P-Net, R-Net và o-Net

#### 1.1 Stage 1: P-Net
Một bức ảnh thường có nhiều hơn một khuôn mặt và các khuôn mặt sẽ có các kích thước khác nhau. Cần một phương thức để có thể nhận dạng được toàn bộ số khuôn mặt. MTCNN sử dụng **resize** ảnh, để tạo ra một loại các bản copy từ ảnh gốc với kích thước khác nhau, từ to đến nhỏ để tạo thành 1 Image Pyramid.

![m](https://images.viblo.asia/f6ff783f-e293-4d3f-bb85-24bc686654c2.png)

Với mỗi một phiên bản copy-resize của ảnh gốc, sử dụng kernel 12x12 pixel và stride = 2 để đi qua toàn bộ bức ảnh, dò tìm khuôn mặt. Vì các bản copies của ảnh gốc có kích thước khác nhau, cho nên mạng có thể dễ dàng nhận biết được các khuôn mặt với kích thước khác nhau, mặc dù chỉ dùng 1 kernel với kích thước cố định (Ảnh to hơn, mặt to hơn; Ảnh nhỏ hơn, mặt nhỏ hơn). Sau đó, đưa những kernels được cắt ra từ trên và truyền qua mạng P-Net (Proposal Network). Kết quả của mạng cho ra một loạt các bounding boxes nằm trong mỗi kernel, mỗi bounding boxes sẽ chứa tọa độ 4 góc để xác định vị trí trong kernel chứa nó (đã được normalize về khoảng từ (0,1)) và điểm confident (Điểm tự tin) tương ứng.

![m](https://images.viblo.asia/62f1c606-72b6-4415-8f25-3424f9309d75.png)

Để loại trừ bớt các bounding boxes trên các bức ảnh và các kernels, sử dụng 2 phương pháp chính là lập mức Threshold confident - nhằm xóa đi các box có mức confident thấp và sử dụng NMS (Non-Maximum Suppression) để xóa các box có tỷ lệ trùng nhau (Intersection Over Union) vượt qua 1 mức threshold tự đặt nào đó. Hình ảnh dưới đây là minh họa cho phép NMS, những box bị trùng nhau sẽ bị loại bỏ và giữ lại 1 box có mức confident cao nhất.

![m](https://images.viblo.asia/336c55ee-e4a4-416b-9f9e-8a757119e9e1.png) 

Sau khi đã xóa bớt các box không hợp lý, chuyển cac tọa độ của các box về với tọa độ gốc của bức ảnh thật. Do tọa độ của box đã được normalize về khoảng (0,1) tương ứng như kernel, cho nên công việc lúc này chỉ là tính toán độ dài và rộng của kernel dựa theo ảnh gốc, sau đó nhân tọa độ đã được normalize của box với kích thước của của kernel và cộng với tọa độ của các góc kernel tương ứng. Kết quả của quá trình trên sẽ là những tọa độ của box tương ứng ở trên ảnh kích thước ban đầu. Cuối cùng, ta sẽ resize lại các box về dạng hình vuông, lấy tọa độ mới của các box và feed vào mạng tiếp theo, mạng R.

#### 1.2 Stage 2: R-Net

![m](https://images.viblo.asia/eaba9895-2505-4ea0-93e7-144296b83b1c.png)

Mạng R (Refine Network) thực hiện các bước như mạng P. Tuy nhiên, mạng còn sử dụng một phương pháp tên là padding, nhằm thực hiện việc chèn thêm các zero-pixels vào các phần thiếu của bounding box nếu bounding box bị vượt quá biên của ảnh. Tất cả các bounding box lúc này sẽ được resize về kích thước 24x24, được coi như 1 kernel và feed vào mạng R. Kết quả sau cũng là những tọa độ mới của các box còn lại và được đưa vào mạng tiếp theo, mạng O.

#### 1.3 Stage 3: O-Net 

![m](https://images.viblo.asia/263800c5-cd50-4c21-913b-8a250353c6fd.png) 

Cuối cùng là mạng O (Output Network), mạng cũng thực hiện tương tự như việc trong mạng R, thay đổi kích thước thành 48x48. Tuy nhiên, kết quả đầu ra của mạng lúc này không còn chỉ là các tọa độ của các box nữa, mà trả về 3 giá trị bao gồm: 4 tọa độ của bounding box (out[0]), tọa độ 5 điểm landmark trên mặt, bao gồm 2 mắt, 1 mũi, 2 bên cánh môi (out[1]) và điểm confident của mỗi box (out[2]). Tất cả sẽ được lưu vào thành 1 dictionary với 3 keys kể trên.

### 2. Tổng quan về Facenet 

Phần tiếp theo là về Face Verification. Nhiệm vụ chính của bài toán này là đánh giá xem ảnh mặt hiện tại có đúng với thông tin, mặt của một người khác đã có trong hệ thống không. 

Quá trình thực hiện: 

- Sử dụng một tập Dataset với rất nhiều các cá thể người khác nhau, mỗi cá thể có một số lượng ảnh nhất định.

- Xây dựng một mạng DNN dùng để làm Feature Extractor cho Dataset trên, kết quả là 1 embedding 128-Dimensions. Trong paper có 2 đại diện mạng là Zeiler&Fergus và InceptionV1.

- Huấn luyện mạng DNN để kết quả embedding có khả năng nhận diện tốt, bao gồm 2 việc là sử dụng l2 normalization (Khoảng cách Euclide) cho các embeddings đầu ra và tối ưu lại các parameters trong mạng màng bằng Triplet Loss.

- Hàm Triplet Loss sẽ sử dụng phương pháp Triplet Selection, lựa chọn các embeddings sao cho việc học diễn ra tốt nhất.

![m](https://images.viblo.asia/920dab46-a78b-4ec3-be72-9da25a367a40.png) 


## Phần 2: Chuẩn bị 


- Trong Dataset/Facenet/raw tạo thư mục (yêu cầu 2 người trở lên), mỗi người 10 bức ảnh bao gồm các biểu cảm: vui, buồn...

- Bước Preprocessing, MTCNN xử lý và đưa ra ảnh đầu ra được lưu trong thư mục peocessed 

    python3 src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

![m](https://github.com/UocNTh/Face-Recognition-with-MTCNN-and-Facenet/blob/main/Image/Screenshot%20from%202023-11-26%2011-35-47.png)


## Phần 3. Tiến hành train model để nhận diện khuôn mặt

    python3 src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000


## Phần 4: Nhận diện khuôn mặt 

    python3 src/face_rec_cam_iot.py 


----
[Tài liệu tham khảo](https://github.com/UocNTh/Face-Recognition-with-MTCNN-and-Facenet/blob/main/Docs/document.pdf)
