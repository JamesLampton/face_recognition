# face_recognition
Toy project for facial recognition.

```bash
docker run --privileged --rm=true -v /path/to/face_recognition/:/state -v /home/.../Pictures/:/home/.../Pictures -ti JamesLampton/dlib /bin/bash
/state/python/face_recognition.py /shape_predictor_68_face_landmarks.dat /dlib_face_recognition_resnet_model_v1.dat /home/.../Pictures/ /state/faces.sqlite3
/state/python/cluster_faces.py /state/faces.sqlite3 /state/clusters/
```
