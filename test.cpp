#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <sstream>
#include <stdio.h>
#include <wiringPi.h>
#include <softPwm.h>
#include <softTone.h>

#define LED_RED 4
#define LED_GREEN 7
#define BUZZER 25
#define TRIG 23
#define ECHO 24

int start_time, end_time;
double dis;

using namespace cv; 
using namespace std;

//1.탐지차량표시
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color, string text);
//2.숫자를 텍스트로 변환(FPS를 텍스트로 출력시) 함수
string intToString(int n); 
//3.차량위험도판단, 선긋기 
//void Detect danger();
//4.차량 레이블
//5.오토바이 탐지
int getMatches(const Mat& Car1, const Mat& Car2, Mat imagematches);
//6.속도탐지

int Danger = 0, BaseLine=220;

int main()
{
    //Video설정
    cv::VideoCapture cap;
    Mat mFrame, mGray, result;
    //분류기 설정
    cv::CascadeClassifier car_cascade; //차량
    if (!car_cascade.load("./car_cascade.xml")) { 
        std::cout << "Error when loading the cascade classfier!" << std::endl; 
        return -1; 
    }   
    cv::CascadeClassifier bike_cascade; //Bike
 	if (!car_cascade.load("./bike_cascade.xml")) {
        std::cout << "Error when loading the cascade classfier!" << std::endl;
        return -1;
    }
    //predict the label of this image
    std::vector<cv::Rect> car_detections;
    std::vector<cv::Rect> bike_detections;
    //영상 읽기 
    cap.open("./RightMode02.mp4");
    //cap.set(CV_CAP_PROP_FPS,10);
	if(wiringPiSetup()==-1)
        {
                puts("setup wiringPi failed!\n");
                return -1;
        }
    while(cap.read(mFrame))
    {
		Danger=0;
		pinMode(LED_GREEN, OUTPUT);
		pinMode(LED_RED, OUTPUT);
		pinMode(BUZZER, OUTPUT);
		pinMode(TRIG, OUTPUT);
		pinMode(ECHO, INPUT);
		digitalWrite(TRIG, LOW);
	    delay(500);
	    digitalWrite(TRIG, HIGH);
	    delayMicroseconds(10);
	    digitalWrite(TRIG, LOW);
	    while(digitalRead(ECHO)==0);
	    start_time=micros();
	    while(digitalRead(ECHO)==1);
	    end_time =  micros();
	    dis = (end_time-start_time)/29./2.;
	    printf("distance : %.2f\n", dis);
	        
		if(dis <=100){Danger=1;}
	
		if(Danger==1)
		{
		softToneCreate(BUZZER);
		digitalWrite(LED_GREEN, 0);
		digitalWrite(LED_RED, 1);
		softToneWrite(BUZZER, 1000);
	    digitalWrite(TRIG, LOW);
	    delay(500);
	    digitalWrite(TRIG, HIGH);
	    delayMicroseconds(10);
	    digitalWrite(TRIG, LOW);
	    while(digitalRead(ECHO)==0);
	    start_time=micros();
	    while(digitalRead(ECHO)==1);
	    end_time =  micros();
	    dis = (end_time-start_time)/29./2.;
		printf("distance : %.2f\n", dis);
	    if(dis <=100){Danger=1;}
		}
		else if(Danger==0)
	{		
		softToneWrite(BUZZER, 0);
		digitalWrite(LED_GREEN, 1);
		digitalWrite(LED_RED, 0);
		digitalWrite(TRIG, LOW);
		delay(500);
		digitalWrite(TRIG, HIGH);
		delayMicroseconds(10);
		digitalWrite(TRIG, LOW);
		while(digitalRead(ECHO)==0);
		start_time=micros();
		while(digitalRead(ECHO)==1);
		end_time =  micros();
		dis = (end_time-start_time)/29./2.;
		printf("distance : %.2f\n", dis);
		if(dis <=100)
			Danger=1;
	}
        //영상반전
        cv::flip(mFrame,result,-1);//수평이면 양수, 수직이면 0 모두라면 음수 
        //그레이로 변환
     	cvtColor(result, mGray, COLOR_BGR2GRAY);
        //cv::Sobel(mGray,SobelX,CV_8U,1,0,3,0.4,128);
        //cv::Sobel(mGray,SobelY,CV_8U,1,0,3,0.4,128);
        //cvtColor(mFrame, mGray, COLOR_BGR2HSV);
        equalizeHist(mGray, mGray);
        //GaussianBlur(mGray, mGray, Size(3,3), 1.5, 1.5);
        cv::Mat mROI = mGray(cv::Rect(Point(mGray.cols/10,mGray.rows/10),
                    Point(mGray.cols*8/10,mGray.rows)));
        //cv::mROI=mFrame(Rect(0, videoSize.height / 2, videoSize.width / 2, videoSize.height / 2
        car_cascade.detectMultiScale(mGray, // input image 
                car_detections, // detection results
                1.1,        // scale reduction factor
                3,          // number of required neighbor detections
                0,          // flags (not used)
                cv::Size(80,80),    // minimum object size to be detected
                cv::Size(480,480)); // maximum object size to be detected
        std::cout << "detections= " << car_detections.size() << std::endl;

        //FPS표시
        int fps=(int)(cap.get(CAP_PROP_FPS));
 		Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int)cap.get(CAP_PROP_FRAME_HEIGHT));
        cout<<"fps="<<fps<<endl;
        cout<<"Size="<<size<<endl;
        int delay=1000/fps;

        //감지한 차량 표시
        draw_locations(result, car_detections, Scalar(0,0,255),"Car");
        draw_locations(result, bike_detections, Scalar(0,0,255),"Bike");
        //정보창 텍스트 입력
        string a = "FPS = ";
        string b = intToString(fps); //FPS
        putText(result,a,Point(5,30),FONT_HERSHEY_PLAIN,1.5,255,2); //FPS
        putText(result,b,Point(85,30),FONT_HERSHEY_PLAIN,1.5,255,2); //FPS
        //BaseLine 선 표시
        line(result,Point(380,180),Point(600,180),Scalar(0,255,0),5);
        //감지한 ROI 표시
        rectangle(result,Point(result.cols/10,result.rows/10),
                Point(result.cols*8/10,result.rows),Scalar(122,122,122),2);
        
		//AKAZE매칭
		//   Mat Car1=imread("/home/terry67867/ca6.PNG",IMREAD_GRAYSCALE);
       // Mat Car2=imread("/home/terry67867/ca7.PNG",IMREAD_GRAYSCALE);
       // Mat Car3=imread("/home/terry67867/ca11.PNG",IMREAD_GRAYSCALE);
       // Mat Car4=imread("/home/terry67867/ca12.PNG",IMREAD_GRAYSCALE);
       // Mat Car5=imread("/home/terry67867/ca17.PNG",IMREAD_GRAYSCALE);
       // Mat Car6=imread("/home/terry67867/ca24.PNG",IMREAD_GRAYSCALE);
       // Mat Car7=imread("/home/terry67867/ca28.PNG",IMREAD_GRAYSCALE);
       // Mat Car8=imread("/home/terry67867/bike10.PNG",IMREAD_GRAYSCALE);
    	//Mat srcImage2=imread("/home/terry67867/cub2_1.jpg",IMREAD_GRAYSCALE);
    	//getMatches(Car1,mROI,result);
       // getMatches(Car2,mROI,result);
       // getMatches(Car3,mROI,result);
       // getMatches(Car4,mROI,result);
       // getMatches(Car5,mGray,result);
       // getMatches(Car6,mGray,result);
       // getMatches(Car7,mGray,result);
       // getMatches(Car8,mGray,result);
        // getMatches(srcImage1,srcImage2);
    cv::imshow("CarDetection", result);

        int ckey=waitKey(delay);
        if(ckey==27)break;

    }
}
//탐지차량표시함수
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color, string text)
{
    int distance=0;
    int r_velocity=0;
    int now_position=0, past_position=200;
    string dis,v;

    if (!locations.empty())
    {
        for( int i = 0 ; i < locations.size() ; i++){
            //탐지차량빨간색사각형
            rectangle(img, locations[i], Scalar(0, 0, 255), 4);
            rectangle(img, Point(locations[i].x-1,locations[i].y),
                    Point(locations[i].x+1+locations[i].width,locations[i].y-15),
                    Scalar(0,0,255),-1);

            //차량글씨
            putText(img, text, Point(locations[i].x-1,locations[i].y-1),
                    FONT_HERSHEY_PLAIN,1, Scalar(0,255,0),1);
            //속도탐지
            now_position = locations[i].x;
            r_velocity = (now_position-past_position)*sqrt(2.0)/3600*24/1000/2000;
            cout<<"r_velocity = " <<r_velocity<<endl;
            // putText(img, dis, Point(locations[i].x,locations[i].y+locations[i].height-5), FONT_HERSHEY_D
            if (text == "Car"){
                cout<<"rectangle(x)= "<<locations[i].x<<endl;
                cout<<"rectangle(y)= "<<locations[i].y<<endl;
                distance = 50*sqrt((locations[i].width*locations[i].width)+
                        (locations[i].height*locations[i].height))
                    /(20);// 2 is avg. width of the car                
            }

  if (text=="Bike"){
                cout<<"rectangle(x)= "<<locations[i].x<<endl;
                cout<<"rectangle(y)= "<<locations[i].y<<endl;
                distance = 50*sqrt((locations[i].width*locations[i].width)+
                        (locations[i].height*locations[i].height))
                    /(20);// 2 is avg. width of the car                

            }
            //위험도판단
            //기준선 위에 있는 경우 차선변경해도 안전 : Danger=0
            //기준선 아래에 있는 경우 차선변경하면 위험 : Danger=1
            if(locations[i].x+50>BaseLine){Danger=1;}
            if(locations[i].x+50<BaseLine){Danger=0;}
	
            cout<<"Danger = "<<Danger<<endl;
            //거리표시
            stringstream stream;
            stream << fixed << setprecision(2) << distance;
            dis = stream.str() + "cm";
            putText(img, dis, Point(locations[i].x-3,locations[i].y+locations[i].height+5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255,255), 1);
            //상대속도표시
            stream << fixed << setprecision(4) << r_velocity;
            v =  stream.str() + "km/h";
            putText(img, v, Point(locations[i].x+30,locations[i].y-1),
                    FONT_HERSHEY_PLAIN,1, Scalar(0,255,0),1);
            past_position = locations[i].x;
            rectangle(img, Point(locations[i].x-1+r_velocity/1000,locations[i].y+r_velocity/1000),
                    Point(locations[i].x+4 + r_velocity/1000 +
                        locations[i].width,locations[i].y+r_velocity/1000),
                    Scalar(0,255,0),-1);
            line(img,Point(locations[i].x-200,locations[i].y+200)
                    ,Point(locations[i].x-50,locations[i].y+200),Scalar(0,0,255),2);
            line(img,Point(locations[i].x,locations[i].y+locations[i].height)
                    ,Point(locations[i].x-200,locations[i].y+200),Scalar(0,255,0),2);
            line(img,Point(locations[i].x+locations[i].width,locations[i].y+locations[i].height)
                    ,Point(locations[i].x-50,locations[i].y+200),Scalar(0,255,0),2);
         }
    }
}
//숫자->문자열변환
string intToString(int n)
{
    stringstream s;
    s << n;
    return s.str();
}

int getMatches(const Mat& Car1, const Mat& Car2, Mat imagematches){
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    //  Ptr<AKAZE> kazeF = AKAZE::create(AKAZE::DESCRIPTOR_KAZE);
    Ptr<AKAZE> kazeF = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
    //  Ptr<AKAZE> kazeF = AKAZE::create(AKAZE::DESCRIPTOR_MLDB); // binary discriptor
    //  Ptr<AKAZE> kazeF = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT); // binary discriptor
    kazeF->detectAndCompute(Car1, noArray(), keypoints1, descriptors1);
    kazeF->detectAndCompute(Car2, noArray(), keypoints2, descriptors2);

    //  Step 3: Matching descriptor vectors
    int k = 2;
    vector< vector< DMatch > > matches;
    Ptr<DescriptorMatcher> matcher;
    //  matcher = DescriptorMatcher::create("BruteForce");
    //  AKAZE::DESCRIPTOR_MLDB, AKAZE::DESCRIPTOR_MLDB_UPRIGHT
    //  matcher = DescriptorMatcher::create("BruteForce-Hamming"); 
    // AKAZE::DESCRIPTOR_KAZE, AKAZE::DESCRIPTOR_KAZE_UPRIGHT
    matcher = DescriptorMatcher::create("FlannBased");

    matcher->knnMatch(descriptors1, descriptors2, matches, k);
    cout << "matches.size()=" << matches.size() << endl;

    vector< DMatch > goodMatches;
    float nndrRatio = 0.6f;
    for (int i = 0; i < matches.size(); i++)
    {
        //      cout << "matches[i].size()=" << matches[i].size() << endl;
        if (matches.at(i).size() == 2 &&
                matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
        {
            goodMatches.push_back(matches[i][0]);

       }
    }
    cout << "goodMatches.size()=" << goodMatches.size() << endl;

    // draw good_matches
    Mat imgMatches;
    drawMatches(Car1, keypoints1, Car2, keypoints2,
           goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); //DEFAULT
    if (goodMatches.size() < 4)
        return 0;

    // find Homography between keypoints1 and keypoints2
    vector<Point2f> obj;
    vector<Point2f> scene;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        // Get the keypoints from the good matches
        obj.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        scene.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }
    Mat H = findHomography(obj, scene, RANSAC); //CV_RANSAC

    vector<Point2f> objP(4);
    objP[0] = Point2f(0, 0);
    objP[1] = Point2f(Car1.cols, 0);
    objP[2] = Point2f(Car1.cols, Car1.rows);
    objP[3] = Point2f(0, Car1.rows);

    vector<Point2f> sceneP(4);
    perspectiveTransform(objP, sceneP, H);

    // draw sceneP in imgMatches
    for (int i = 0; i < 4; i++)

      sceneP[i] += Point2f(Car1.cols, 0);
    for (int i = 0; i < 4; i++)
        line(imagematches, sceneP[i], sceneP[(i + 1) % 4], Scalar(255, 0, 0), 4);
}

