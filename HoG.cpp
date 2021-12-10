#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <string.h>

using namespace cv;
using namespace std;
using namespace dnn;

// explaining HoG
// h: HoG descriptor
// H: HoG descriptor on a foregroundMask
// f: findContours
// y: YOLO

int main() {
    Mat frame, gray, foregroundMask, foregroundImg, avg, background;
    VideoCapture capture("background.mp4");
    VideoCapture cap("background.mp4");
    int cnt = 2;

    cap >> avg;

    while (cnt <= 11) {
        if(!cap.read(frame)) break;
        cvtColor(avg, background, CV_BGR2GRAY);
        add(frame/cnt, avg*(cnt-1)/cnt, avg);
        cnt++;
    }

    /* HoG parameters */
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    /* YOLO parameters */
    String modelConfiguration = "deep/yolov2-tiny.cfg";
    String modelBinary = "deep/yolov2-tiny.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelBinary);

    char ch;
    while (1) 
    {
        if (capture.grab()==0) break;
        capture.retrieve(frame);
        cvtColor(frame, gray, CV_BGR2GRAY);
        absdiff(background, gray, foregroundMask);
        threshold(foregroundMask, foregroundMask, 20, 255, CV_THRESH_BINARY);
        foregroundMask.copyTo(foregroundImg);
        gray.copyTo(foregroundImg, foregroundMask);

        char tmp = waitKey(33);
        if (tmp == ch)
            ch = 0;
        else if(tmp > 0)
            ch = tmp;
        else if (ch == 27) break; // ESC Key (exit) 
        else if (ch == 32)
        {// SPACE Key (pause)
            while ((ch = waitKey(33)) != 32 && ch != 27);
            if (ch == 27) break; 
        }
        else if (ch == 104) 
        {// h
            vector<Rect> found;
            cnt = 0;
            hog.detectMultiScale(frame, found);
            for (Rect r : found) {
                Scalar c = Scalar(255, 255, 255);
                rectangle(frame, r, c, 1);
                cnt++;
            }
            putText(frame, format("object count: %d", cnt), Point(50, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 4);
            imshow("frame", frame);
        }
        else if (ch == 72)
        {// H
            vector<Rect> found;
            hog.detectMultiScale(foregroundMask, found);
            cnt = 0;
            for (Rect r : found) {
                Scalar c = Scalar(255, 255, 255);
                rectangle(foregroundMask, r, c, 1);
                cnt++;
            }
            putText(foregroundMask, format("object count: %d", cnt), Point(50, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 4);
            imshow("frame", foregroundMask);
        }
        else if (ch == 102)
        {// f
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            findContours(foregroundMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            cnt=0;
            vector<Rect> boundRect(contours.size());
            for (int i = 0; i < contours.size(); i++) {
                if (contourArea(contours[i])>=400) {
                    cnt++;
                    boundRect[i] = boundingRect(Mat(contours[i]));
                }
            }
            for (int i = 0; i < contours.size(); i++) {
                rectangle(frame, boundRect[i],  Scalar(255, 255, 255), 3, 8, 0);
            }
            putText(frame, format("object count: %d", cnt), Point(50, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 4);
            imshow("frame", frame);
        }
        else if (ch == 121)
        {// y
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
            cnt = 0;
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
                    Scalar object_roi_color = Scalar(255, 255, 255);
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
            putText(frame, format("object count: %d", cnt), Point(50, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 4);
            imshow("frame", frame);
        }
        else
            imshow("frame", frame);
    }
}