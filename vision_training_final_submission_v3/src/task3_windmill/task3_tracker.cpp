#include "common/vision_common.hpp"
#include "task3_windmill/task3_config.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace cv;
using namespace std;

struct Model {
    Point2f center;
    float radius{};
    vector<int> inliers;
    int support{};
    int angular_bins{};
};

static bool fitCircle3(const Point2f& p1, const Point2f& p2, const Point2f& p3,
                       Point2f& c, float& r) {
    double d = 2.0 * (p1.x*(p2.y-p3.y) + p2.x*(p3.y-p1.y) + p3.x*(p1.y-p2.y));
    if (std::abs(d) < 1e-7) return false;
    double ux = ((p1.x*p1.x+p1.y*p1.y)*(p2.y-p3.y)
               + (p2.x*p2.x+p2.y*p2.y)*(p3.y-p1.y)
               + (p3.x*p3.x+p3.y*p3.y)*(p1.y-p2.y))/d;
    double uy = ((p1.x*p1.x+p1.y*p1.y)*(p3.x-p2.x)
               + (p2.x*p2.x+p2.y*p2.y)*(p1.x-p3.x)
               + (p3.x*p3.x+p3.y*p3.y)*(p2.x-p1.x))/d;
    c = Point2f((float)ux,(float)uy);
    r = (float)((norm(c-p1)+norm(c-p2)+norm(c-p3))/3.0);
    return std::isfinite(r);
}

static vector<Model> findModels(const vector<Vec3f>& circles) {
    vector<Model> all;
    const float rmin=55.f, rmax=135.f, tol=12.f;
    for (int a=0;a<(int)circles.size();++a)
      for (int b=a+1;b<(int)circles.size();++b)
       for (int c=b+1;c<(int)circles.size();++c) {
        Point2f ctr; float rad;
        if (!fitCircle3(Point2f(circles[a][0],circles[a][1]),
                         Point2f(circles[b][0],circles[b][1]),
                         Point2f(circles[c][0],circles[c][1]),ctr,rad)) continue;
        if (rad<rmin || rad>rmax) continue;
        Model m; m.center=ctr; m.radius=rad;
        vector<int> bins;
        for (int i=0;i<(int)circles.size();++i) {
            Point2f p(circles[i][0],circles[i][1]);
            float err=std::abs((float)norm(p-ctr)-rad);
            if (err<tol) {
                m.inliers.push_back(i);
                float a=atan2(p.y-ctr.y,p.x-ctr.x);
                int bin=(int)floor((a+CV_PI)/(2*CV_PI)*12.0);
                bins.push_back(bin);
            }
        }
        sort(bins.begin(),bins.end());
        bins.erase(unique(bins.begin(),bins.end()),bins.end());
        m.support=(int)m.inliers.size();
        m.angular_bins=(int)bins.size();
        if (m.support>=4 && m.angular_bins>=3) all.push_back(m);
       }
    sort(all.begin(),all.end(),[](const Model& a,const Model& b){
        if(a.support!=b.support) return a.support>b.support;
        return a.angular_bins>b.angular_bins;
    });
    vector<Model> out;
    for (auto &m: all) {
        bool dup=false;
        for(auto &q:out) if(norm(m.center-q.center)<12.f && abs(m.radius-q.radius)<12.f){dup=true;break;}
        if(!dup) out.push_back(m);
        if(out.size()>=12) break;
    }
    return out;
}

int main(int argc,char**argv){
    if(argc<3){
        cerr<<"Usage: task3_tracker <input.mp4> <output.mp4> [csv]\n";
        return 1;
    }
    VideoCapture cap(argv[1]);
    if(!cap.isOpened()){cerr<<"Cannot open input\n";return 2;}
    int w=(int)cap.get(CAP_PROP_FRAME_WIDTH), h=(int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps=cap.get(CAP_PROP_FPS); if(fps<=0) fps=30;
    int n=(int)cap.get(CAP_PROP_FRAME_COUNT);
    VideoWriter writer(argv[2],VideoWriter::fourcc('m','p','4','v'),fps,Size(w,h));
    ofstream csv(argc>=4?argv[3]:"tracking.csv");
    csv<<"frame,time_s,cx,cy,radius,status,target_id,num_inliers\n";

    Point2f center; Point2f velocity(0,0); float radius=90; bool have=false;
    int lost=0,id=1,reselects=0; double bladeAngle=std::numeric_limits<double>::quiet_NaN();

    for(int f=0;f<n;++f){
        Mat frame;
        if(!cap.read(frame)) break;
        Mat hsv,mask;
        cvtColor(frame,hsv,COLOR_BGR2HSV);
        inRange(hsv,Scalar(0,45,35),Scalar(40,255,255),mask);
        morphologyEx(mask,mask,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));

        Mat blur; GaussianBlur(mask,blur,Size(7,7),1.5);
        vector<Vec3f> circles;
        HoughCircles(blur,circles,HOUGH_GRADIENT,1.0,22,70,12,12,38);
        auto models=findModels(circles);

        Model chosen; bool got=false; string status="lost";
        vector<int> good;
        for(int i=0;i<(int)models.size();++i) good.push_back(i);

        if(!have){
            if(!good.empty()){
                chosen=models[good[0]]; got=true;
                center=chosen.center; radius=chosen.radius; velocity=Point2f(0,0);
                have=true; lost=0; status="detected";
            }
        } else {
            Point2f pred=center+velocity;
            double best=1e18;
            for(int idx:good){
                auto &m=models[idx];
                double d=norm(m.center-pred);
                if(d>55) continue;
                double cost=d+0.2*abs(m.radius-radius)-2.0*m.support-0.3*m.angular_bins;
                if(cost<best){best=cost;chosen=m;got=true;}
            }
            if(got){
                Point2f nv=chosen.center-center;
                velocity=0.8*velocity+0.2*nv;
                center=0.6*center+0.4*chosen.center;
                radius=0.75f*radius+0.25f*chosen.radius;
                lost=0; status="detected";
            }else{
                ++lost; center=pred; velocity*=0.92f;
                if(lost>60 && !good.empty()){
                    chosen=models[good[0]]; got=true;
                    center=chosen.center; radius=chosen.radius; velocity=Point2f(0,0);
                    lost=0; ++id; ++reselects; status="reselected";
                }
            }
        }

        if(have){
            drawMarker(frame,Point2f(center.x,center.y),Scalar(0,255,0),MARKER_CROSS,18,2);
            putText(frame,"R center",Point2f(center.x+8,center.y-10),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0),1);

            if(got && !chosen.inliers.empty()){
                vector<Point2f> pts;
                for(int k:chosen.inliers){
                    Point2f p(circles[k][0],circles[k][1]); pts.push_back(p);
                    circle(frame,p,(int)circles[k][2],Scalar(255,180,0),1);
                    circle(frame,p,3,Scalar(255,180,0),FILLED);
                }
                int bestK=0;
                if(std::isfinite(bladeAngle)){
                    double bd=1e9;
                    for(int j=0;j<(int)pts.size();++j){
                        double a=atan2(pts[j].y-center.y,pts[j].x-center.x);
                        double d=atan2(sin(a-bladeAngle),cos(a-bladeAngle));
                        if(abs(d)<bd){bd=abs(d);bestK=j;}
                    }
                }else{
                    for(int j=1;j<(int)pts.size();++j)
                        if(pts[j].y<pts[bestK].y) bestK=j;
                }
                Point2f bp=pts[bestK];
                bladeAngle=atan2(bp.y-center.y,bp.x-center.x);
                circle(frame,bp,(int)circles[chosen.inliers[bestK]][2],Scalar(0,0,255),2);
                circle(frame,bp,4,Scalar(0,0,255),FILLED);
                line(frame,center,bp,Scalar(0,255,255),2);
            }
            Scalar tc=status=="lost"?Scalar(0,165,255):Scalar(0,255,0);
            putText(frame,format("ID %d  %s",id,status.c_str()),Point(15,28),FONT_HERSHEY_SIMPLEX,0.7,tc,2);
            putText(frame,format("lost=%d",lost),Point(15,53),FONT_HERSHEY_SIMPLEX,0.55,tc,1);
        }else{
            putText(frame,"ID 0  waiting",Point(15,28),FONT_HERSHEY_SIMPLEX,0.7,Scalar(0,165,255),2);
        }

        writer.write(frame);
        csv<<f<<","<<f/fps<<",";
        if(have) csv<<center.x<<","<<center.y<<","<<radius;
        else csv<<",,";
        csv<<","<<status<<","<<id<<","<<(got?chosen.support:0)<<"\n";
    }
    writer.release(); cap.release(); csv.close();
    cerr<<"reselects="<<reselects<<"\n";
    return 0;
}
