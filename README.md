In this project, we developed a robust face recognition system to accurately identify individuals in a diverse set of images. The primary goal was to build a system that can effectively recognize multiple faces in a single image, handling variations in lighting, facial expressions, and poses. To achieve this, we employed the widely-used face_recognition library, which is built on top of the state-of-the-art dlib library. Our approach began with encoding the known faces from a curated dataset, ensuring a variety of images with different facial expressions, poses, and lighting conditions. This dataset was used to train the model to recognize the individuals in the images. We then tested the model on a separate set of images containing multiple faces, evaluating its ability to detect and identify each face accurately. Throughout the project, we made several improvements to enhance the model's accuracy. These included adjusting the tolerance value for face comparison, optimizing image preprocessing techniques, and refining the face encoding process. As a result of these adjustments, we achieved a significant improvement in the model's recognition accuracy. The highlights of our key findings include the model's capability to recognize multiple faces in a single image, its robustness against variations in lighting, and its ability to identify individuals with diverse facial expressions and poses. Overall, our face recognition system demonstrates strong potential for practical applications, such as access control, surveillance, and social media tagging, while maintaining user-friendly implementation and adaptability to various datasets.