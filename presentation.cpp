#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
// 22000217 나예원
using namespace cv;
using namespace std;
using namespace dnn;

Mat background_subtractor(Mat frame, Mat foregroundImg, Net net);
void face_detector(Mat frame, Mat grayframe, CascadeClassifier face_classifier);
Mat grab_cut(Mat frame, vector<Rect> faces);

int main() {
    VideoCapture capture("Faces.mp4");
    Mat frame, gray, foregroundMask, foregroundImg;

    /* HoG parameters */
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    /* detecting people in video */
    String modelConfiguration = "deep/yolov2-tiny.cfg";
    String modelBinary = "deep/yolov2-tiny.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelBinary);

    /* background subtraction */
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
    capture.set(CV_CAP_PROP_POS_FRAMES,0);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

    /* face detection */
    CascadeClassifier face_classifier;

    /* virtual background */
    Mat foreground, result, bgModel, fgModel;

    char ch;

    while(true){

        if (capture.get(CV_CAP_PROP_FRAME_COUNT) - 5 == capture.get(CV_CAP_PROP_POS_FRAMES)) {
            capture.set(CV_CAP_PROP_POS_FRAMES, 2);
        }

        if(capture.grab() == 0)
            break;
        
        capture.retrieve(frame);
        resize(frame, frame, Size(frame.cols/1.2, frame.rows/1.2), 0, 0, INTER_LINEAR);
        char tmp = waitKey(33);
        if (tmp == ch)
            ch = 0;
        else if(tmp > 0)
            ch = tmp;

        if (ch == 27) break; // ESC Key (exit) 
        else if (ch == 32){ // SPACE Key (pause)
            while ((ch = waitKey(33)) != 32 && ch != 27);
            if (ch == 27) break; 
        }
        else if (ch == 98) {
            /* subtracts background and shows only the foreground information */
            pBackSub->apply(frame, foregroundMask);
            threshold(foregroundMask, foregroundMask, 20, 255, CV_THRESH_BINARY);
            GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
            morphologyEx(foregroundMask, foregroundMask, MORPH_CLOSE, element);
            foregroundMask.copyTo(foregroundImg);
            frame.copyTo(foregroundImg, foregroundMask);
            frame = background_subtractor(frame, foregroundImg, net).clone();
            imshow("result", frame);
        }
        else if (ch == 102) {
            /* face detection */
            face_detector(frame, gray, face_classifier);
            imshow("result", frame);
        }
        else if (ch == 103) {
            vector<Rect> faces;
            face_classifier.load("haarcascade_frontalface_alt.xml");
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            face_classifier.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
            frame = grab_cut(frame, faces).clone();
            imshow("result", frame);
        }
        else if (ch == 104) 
        { // HOG
            vector<Rect> found;
            int cnt = 0;
            hog.detectMultiScale(frame, found);
            for (Rect r : found) {
                Scalar c = Scalar(0, 255, 0);
                rectangle(frame, r, c, 2);
                cnt++;
            }
            putText(frame, format("There are %d people.", cnt), Point(50, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 4);
            imshow("result", frame);

        }
        else if (ch == 121)
        {// YOLO
            vector<String> classNamesVec;
            ifstream classNamesFile("deep/coco.names");
            if(classNamesFile.is_open()) {
                string className = "";
                while(getline(classNamesFile, className)) {
                    classNamesVec.push_back(className);
                }
            }
            if (frame.channels() == 4) cvtColor(frame, frame, CV_BGR2GRAY);
            Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
            net.setInput(inputBlob, "data");
            Mat detection = net.forward("detection_out");

            float confidenceThreshold = 0.5;
           int cnt = 0;
            for (int i=0; i<detection.rows; i++) {
                const int probability_index = 5;
                const int probability_size = detection.cols - probability_index;
                float *prob_array_ptr = &detection.at<float>(i, probability_index);
                size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
                float confidence = detection.at<float>(i, (int)objectClass + probability_index);
                if (confidence > confidenceThreshold) {
                    float x_center = detection.at<float>(i, 0) * frame.cols;
                    float y_center = detection.at<float>(i, 1) * frame.rows;
                    float width = detection.at<float>(i, 2) * frame.cols;
                    float height = detection.at<float>(i, 3) * frame.rows;
                    Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                    Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                    Rect object(p1, p2);
                    Scalar object_roi_color = Scalar(0, 255, 0);
                    rectangle(frame, object, object_roi_color, 2);
                    String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
                    String label = format("%s: %.2f", className.c_str(), confidence);
                    int baseLine = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    rectangle(frame, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)), object_roi_color, FILLED);
                    putText(frame, label, p1 + Point(0, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                    cnt++;
                }
            }
            putText(frame, format("There are %d people.", cnt), Point(50, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 4);
            imshow("result", frame);
        }
        else
            imshow("result", frame);
    }
}

Mat background_subtractor(Mat frame, Mat foregroundImg, Net net) {
    vector<String> classNamesVec;
    ifstream classNamesFile("deep/coco.names");
    if(classNamesFile.is_open()) {
        string className = "";
        while(getline(classNamesFile, className)) {
            classNamesVec.push_back(className);
        }
    }
    if (frame.channels() == 4) cvtColor(frame, frame, CV_BGR2GRAY);
    Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");

    float confidenceThreshold = 0.5;
    Mat black(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    int human_count = 0;

    for (int i=0; i<detection.rows; i++) {
        const int probability_index = 5;
        const int probability_size = detection.cols - probability_index;
        float *prob_array_ptr = &detection.at<float>(i, probability_index);
        size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detection.at<float>(i, (int)objectClass + probability_index);
        if (confidence > confidenceThreshold) {
            float x_center = detection.at<float>(i, 0) * frame.cols;
            float y_center = detection.at<float>(i, 1) * frame.rows;
            float width = detection.at<float>(i, 2) * frame.cols;
            float height = detection.at<float>(i, 3) * frame.rows;
            String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
            if (className.c_str()[0] == 'p') {
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                Rect object(p1, p2);
                if (object.y + object.height > frame.rows) object.height = frame.rows - object.y;
                for (int x = object.x; x < object.x + object.width; x++) {
                    for (int y = object.y; y < object.y + object.height; y++) {
                        black.at<Vec3b>(y, x) = frame.at<Vec3b>(y, x);
                    }
                }
                bitwise_and(foregroundImg, black, black);
                Scalar object_roi_color = Scalar(255, 255, 255);    // white, rectangle and text color around the detected obejct
                rectangle(frame, object, object_roi_color, 2); // drawing rectangle at detected object
                human_count++;
            }
        }
    }
    String human_label = format("There are %d people.", human_count);
    putText(black, human_label, Point(50,60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 4);
    return black;
}

void face_detector(Mat frame, Mat gray, CascadeClassifier face_classifier) {
    vector<Rect> faces;
    face_classifier.load("haarcascade_frontalface_alt.xml");
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    face_classifier.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
    Scalar s1 = Scalar(0, 255, 0);  // green
    Scalar s2 = Scalar(0, 255, 255);    //yellow
    Scalar s3 = Scalar(255, 0, 0);  //blue
    int sizes[3];
    for (int i=0; i<3; i++) {
        sizes[i] = faces[i].width * faces[i].height;
    }
    sort(sizes, sizes+3);
    for (int i = 0; i < faces.size(); i++) {
        Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        Point tr(faces[i].x, faces[i].y);
        if (faces[i].width * faces[i].height == sizes[0]) {
            rectangle(frame, tr, lb, s3, 2);
        }
        else if (faces[i].width * faces[i].height == sizes[1]) {
            rectangle(frame, tr, lb, s2, 2);
        }
        else if (faces[i].width * faces[i].height == sizes[2]) {
            rectangle(frame, lb, tr, s1, 3, 4, 0);
        }
    }
}

Mat grab_cut(Mat frame, vector<Rect> faces) {
    Mat temp, result, bgModel, fgModel;
    Mat img = imread("image.jpg");
    resize(img, img, Size(frame.cols, frame.rows), 0, 0, INTER_LINEAR);
    grabCut(frame, result, faces[0], bgModel, fgModel, 1, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);
    for (int i = 1; i < faces.size(); i++) {
        grabCut(frame, temp, faces[i], bgModel, fgModel, 1, GC_INIT_WITH_RECT);
        compare(temp, GC_PR_FGD, temp, CMP_EQ);
        for (int x = 0; x < result.cols; x++) {
            for (int y = 0; y < result.rows; y++) {
                if (temp.at<uchar>(y,x) == 255) {
                    result.at<uchar>(y,x) = 255;
                }
            }
        }
    }
    frame.copyTo(img,result);
    return img;
}
