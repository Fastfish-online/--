#include "common/vision_common.hpp"
#include "task2_fit/task2_model.hpp"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <filesystem>

using namespace cv;
using namespace std;

struct Obs {
    int frame;
    double t, x, y, theta;
};

static double wrapPi(double x) {
    while (x >= CV_PI) x -= 2.0 * CV_PI;
    while (x < -CV_PI) x += 2.0 * CV_PI;
    return x;
}

static double thetaModel(const Eigen::VectorXd& p, double t) {
    const double th0=p(0), b=p(1), A=p(2), O=p(3), phi=p(4);
    return th0 + b*t + (A/O)*(std::cos(phi)-std::cos(O*t+phi));
}

// Draw a simple line plot using OpenCV only.
static void savePlot(const string& path, const vector<double>& xs,
                     const vector<double>& ys1, const vector<double>& ys2,
                     const string& title, const string& yLabel,
                     const string& legend1, const string& legend2,
                     bool zeroLine=false) {
    const int W=1200, H=650, L=90, R=35, T=65, B=75;
    Mat img(H,W,CV_8UC3,Scalar(255,255,255));
    double xmin=*min_element(xs.begin(),xs.end()), xmax=*max_element(xs.begin(),xs.end());
    double ymin=1e100,ymax=-1e100;
    for(double v:ys1){ymin=min(ymin,v);ymax=max(ymax,v);}
    if(!ys2.empty()) for(double v:ys2){ymin=min(ymin,v);ymax=max(ymax,v);}
    if (zeroLine){ymin=min(ymin,0.0);ymax=max(ymax,0.0);}
    if (fabs(ymax-ymin)<1e-12){ymin-=1;ymax+=1;}
    double pad=0.06*(ymax-ymin); ymin-=pad; ymax+=pad;
    auto px=[&](double x){return L+(int)llround((x-xmin)/(xmax-xmin)*(W-L-R));};
    auto py=[&](double y){return H-B-(int)llround((y-ymin)/(ymax-ymin)*(H-T-B));};
    line(img,Point(L,T),Point(L,H-B),Scalar(40,40,40),2);
    line(img,Point(L,H-B),Point(W-R,H-B),Scalar(40,40,40),2);
    if(zeroLine && ymin<0 && ymax>0) line(img,Point(L,py(0)),Point(W-R,py(0)),Scalar(180,180,180),1);
    putText(img,title,Point(30,38),FONT_HERSHEY_SIMPLEX,0.9,Scalar(20,20,20),2);
    putText(img,"Time (s)",Point(W/2-45,H-20),FONT_HERSHEY_SIMPLEX,0.65,Scalar(20,20,20),1);
    putText(img,yLabel,Point(12,H/2),FONT_HERSHEY_SIMPLEX,0.62,Scalar(20,20,20),1);
    // ticks
    for(int k=0;k<=6;k++){
        double x=xmin+(xmax-xmin)*k/6.0; int X=px(x);
        line(img,Point(X,H-B),Point(X,H-B+6),Scalar(50,50,50),1);
        putText(img,format("%.1f",x),Point(X-18,H-B+28),FONT_HERSHEY_SIMPLEX,0.45,Scalar(50,50,50),1);
    }
    for(int k=0;k<=5;k++){
        double y=ymin+(ymax-ymin)*k/5.0; int Y=py(y);
        line(img,Point(L-6,Y),Point(L,Y),Scalar(50,50,50),1);
        putText(img,format("%.3f",y),Point(5,Y+5),FONT_HERSHEY_SIMPLEX,0.42,Scalar(50,50,50),1);
    }
    auto drawSeries=[&](const vector<double>& ys, Scalar c, int thick){
        for(size_t i=1;i<xs.size();++i) line(img,Point(px(xs[i-1]),py(ys[i-1])),Point(px(xs[i]),py(ys[i])),c,thick,LINE_AA);
    };
    drawSeries(ys1,Scalar(40,100,220),2);
    if(!ys2.empty()) drawSeries(ys2,Scalar(60,160,60),2);
    rectangle(img,Point(W-330,20),Point(W-20,55+(ys2.empty()?0:25)),Scalar(230,230,230),FILLED);
    line(img,Point(W-315,38),Point(W-280,38),Scalar(40,100,220),3);
    putText(img,legend1,Point(W-270,43),FONT_HERSHEY_SIMPLEX,0.48,Scalar(30,30,30),1);
    if(!ys2.empty()){
        line(img,Point(W-315,63),Point(W-280,63),Scalar(60,160,60),3);
        putText(img,legend2,Point(W-270,68),FONT_HERSHEY_SIMPLEX,0.48,Scalar(30,30,30),1);
    }
    imwrite(path,img);
}

int main(int argc,char** argv){
    string input = argc>1 ? argv[1] : "resources/task_2.mp4";
    string outDir = argc>2 ? argv[2] : "result/task2_fit";
    std::filesystem::create_directories(outDir);
    VideoCapture cap(input);
    if(!cap.isOpened()){cerr<<"Cannot open "<<input<<"\n"; return 1;}
    double fps=cap.get(CAP_PROP_FPS);
    int n=(int)cap.get(CAP_PROP_FRAME_COUNT);
    int W=(int)cap.get(CAP_PROP_FRAME_WIDTH), H=(int)cap.get(CAP_PROP_FRAME_HEIGHT);
    if(fps<=0 || n<=0){cerr<<"Invalid video properties\n";return 1;}

    const Point2d center(W/2.0,H/2.0);
    vector<Obs> obs; vector<Mat> frames;
    int i=0;
    while(true){
        Mat frame; if(!cap.read(frame)) break;
        Mat hsv,mask;
        cvtColor(frame,hsv,COLOR_BGR2HSV);
        inRange(hsv,Scalar(80,120,80),Scalar(110,255,255),mask);
        vector<vector<Point>> contours;
        findContours(mask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        if(contours.empty()){i++; continue;}
        int best=-1; double bestArea=0;
        for(int j=0;j<(int)contours.size();++j){
            double a=contourArea(contours[j]);
            if(a>bestArea){bestArea=a;best=j;}
        }
        if(best<0 || bestArea<5){i++;continue;}
        Moments m=moments(contours[best]);
        if(fabs(m.m00)<1e-9){i++;continue;}
        double x=m.m10/m.m00, y=m.m01/m.m00;
        double th=atan2(center.y-y,x-center.x);
        obs.push_back({i,i/fps,x,y,th});
        frames.push_back(frame.clone());
        i++;
    }
    if(obs.size()<20){cerr<<"Too few valid observations\n";return 1;}

    // Unwrap angle.
    for(size_t k=1;k<obs.size();++k){
        double d=obs[k].theta-obs[k-1].theta;
        while(d>CV_PI) d-=2*CV_PI;
        while(d<-CV_PI) d+=2*CV_PI;
        obs[k].theta=obs[k-1].theta+d;
    }

    // Levenberg-Marquardt fit of integrated angular-velocity model.
    // Several initial frequencies/phases are tried because the nonlinear model
    // has local minima; the best constrained solution is retained.
    auto cost=[&](const Eigen::VectorXd& q){
        double s=0; for(auto&o:obs){double r=o.theta-thetaModel(q,o.t);s+=r*r;} return s;
    };
    Eigen::VectorXd bestP(5);
    double bestCost=1e300;
    const double meanRate=(obs.back().theta-obs.front().theta)/(obs.back().t-obs.front().t);
    for(double O0: {0.55,0.85,1.15,1.45,1.65,1.95,2.35}) {
        for(double phi0: {-CV_PI,-1.5,-0.5,0.5,1.5,2.5}) {
            Eigen::VectorXd p(5);
            p << obs.front().theta, meanRate, 0.55, O0, phi0;
            double lambda=1e-3;
            double prev=cost(p);
            for(int it=0;it<500;++it){
                Eigen::MatrixXd J(obs.size(),5);
                Eigen::VectorXd r(obs.size());
                for(size_t k=0;k<obs.size();++k){
                    double tt=obs[k].t, O=p(3), A=p(2), ph=p(4), q=O*tt+ph;
                    double cphi=cos(ph), cq=cos(q), sq=sin(q);
                    J(k,0)=1.0;
                    J(k,1)=tt;
                    J(k,2)=(cphi-cq)/O;
                    J(k,3)=A*(-(cphi-cq)/(O*O) + tt*sq/O);
                    J(k,4)=A/O*(-sin(ph)+sq);
                    r(k)=obs[k].theta-thetaModel(p,tt);
                }
                Eigen::MatrixXd Hm=J.transpose()*J;
                Eigen::VectorXd g=J.transpose()*r;
                for(int d=0;d<5;++d) Hm(d,d)*=(1.0+lambda);
                Eigen::VectorXd dp=Hm.ldlt().solve(g);
                if(!dp.allFinite()) break;
                Eigen::VectorXd trial=p+dp;
                trial(4)=wrapPi(trial(4));
                if(trial(2)<=1e-6 || trial(3)<=1e-4 || trial(1)<=trial(2)) { lambda*=10; continue; }
                double c=cost(trial);
                if(c<prev){p=trial;prev=c;lambda=max(lambda*0.5,1e-9);if(dp.norm()<1e-11)break;}
                else lambda*=5;
            }
            if(p(2)>0 && p(1)>p(2) && p(3)>0 && prev<bestCost){ bestCost=prev; bestP=p; }
        }
    }
    Eigen::VectorXd p=bestP;

    vector<double> ts, yobs, yfit, residual, omegaObs, omegaFit;
    for(auto&o:obs){ts.push_back(o.t);yobs.push_back(o.theta);yfit.push_back(thetaModel(p,o.t));residual.push_back(o.theta-thetaModel(p,o.t));}
    omegaObs.resize(obs.size());
    for(size_t k=0;k<obs.size();++k){
        if(k==0) omegaObs[k]=(yobs[1]-yobs[0])/((obs[1].t-obs[0].t));
        else if(k+1==obs.size()) omegaObs[k]=(yobs[k]-yobs[k-1])/(obs[k].t-obs[k-1].t);
        else omegaObs[k]=(yobs[k+1]-yobs[k-1])/(obs[k+1].t-obs[k-1].t);
    }
    // Moving-average smoothing for diagnostic angular velocity curve.
    vector<double> smooth=omegaObs; int win=21, half=win/2;
    for(size_t k=0;k<omegaObs.size();++k){double s=0;int c=0;for(int j=max<int>(0,k-half);j<=min<int>(omegaObs.size()-1,k+half);++j){s+=omegaObs[j];c++;}smooth[k]=s/c;}
    omegaObs=smooth;
    omegaFit.resize(obs.size());
    for(size_t k=0;k<obs.size();++k) omegaFit[k]=p(1)+p(2)*sin(p(3)*obs[k].t+p(4));

    double sse=0, osse=0;
    for(double r:residual)sse+=r*r;
    for(size_t k=0;k<omegaObs.size();++k){double r=omegaObs[k]-omegaFit[k];osse+=r*r;}
    double rmse=sqrt(sse/residual.size()), omegaRmse=sqrt(osse/omegaObs.size());

    // Save tracking overlay.
    string videoOut=outDir+"/tracking_overlay.mp4";
    VideoWriter vw(videoOut,VideoWriter::fourcc('m','p','4','v'),fps,Size(W,H));
    for(size_t k=0;k<frames.size();++k){
        Mat vis=frames[k].clone();
        circle(vis,center,6,Scalar(255,255,255),FILLED);
        putText(vis,"R",Point((int)center.x+10,(int)center.y-10),FONT_HERSHEY_SIMPLEX,.65,Scalar(255,255,255),2);
        Point2d q(obs[k].x,obs[k].y);
        circle(vis,q,14,Scalar(255,255,0),2); circle(vis,q,4,Scalar(255,255,0),FILLED);
        line(vis,center,q,Scalar(255,255,0),1);
        putText(vis,"ID: 1  detected",Point(15,30),FONT_HERSHEY_SIMPLEX,.75,Scalar(0,255,0),2);
        putText(vis,format("t=%.2fs  theta=%.3f rad",obs[k].t,obs[k].theta),Point(15,58),FONT_HERSHEY_SIMPLEX,.62,Scalar(255,255,255),2);
        vw.write(vis);
    }
    vw.release();

    savePlot(outDir+"/fit_comparison.png",ts,yobs,yfit,"Observed vs fitted angle","Angle (rad)","Observed","Fitted");
    savePlot(outDir+"/angular_velocity.png",ts,omegaObs,omegaFit,"Angular velocity","Angular velocity (rad/s)","Estimated","Model");
    vector<double> empty;
    savePlot(outDir+"/residuals.png",ts,residual,empty,"Angle residuals","Residual (rad)","Residual","",true);

    ofstream md(outDir+"/../task2_fit_result.md");
    md<<fixed<<setprecision(6);
    md<<"# Task 2 - 合成旋转视频参数拟合\n\n";
    md<<"## 输入视频\n";
    md<<"- 文件：`resources/task_2.mp4`\n- 分辨率："<<W<<" x "<<H<<"\n- FPS："<<fps<<"\n- 帧数："<<n<<"\n- 时长："<<n/fps<<" s\n";
    md<<"- 检测中心：("<<center.x<<", "<<center.y<<")（使用视频画面中心；原任务给定中心按实际分辨率等比例对应）\n";
    md<<"- 有效样本："<<obs.size()<<"，帧范围 "<<obs.front().frame<<"–"<<obs.back().frame<<"\n\n";
    md<<"## 方法\n";
    md<<"HSV 提取青色目标，取最大有效连通轮廓质心；按 `atan2(cy-y, x-cx)` 计算从右方起算、逆时针为正的角度，并进行角度展开。直接拟合积分后的角度模型：\n\n";
    md<<"`theta(t)=theta0+b*t+(A/Omega)*(cos(phi)-cos(Omega*t+phi))`\n\n";
    md<<"其导数对应讲义给定的 `omega(t)=b+A*sin(Omega*t+phi)`。使用带约束的 Levenberg-Marquardt 迭代估计参数。\n\n";
    md<<"## 拟合参数\n";
    md<<"| 参数 | 数值 | 单位 |\n|---|---:|---|\n";
    md<<"| theta0 | "<<p(0)<<" | rad |\n| b | "<<p(1)<<" | rad/s |\n| A | "<<p(2)<<" | rad/s |\n| Omega | "<<p(3)<<" | rad/s |\n| phi | "<<wrapPi(p(4))<<" | rad |\n";
    md<<"\n约束检查： A > 0、b > A、Omega > 0。速度变化周期 `T=2*pi/Omega` = "<<2*CV_PI/p(3)<<" s。\n\n";
    md<<"## 误差\n";
    md<<"- 角度 RMSE："<<rmse<<" rad\n";
    md<<"- 角速度诊断 RMSE："<<omegaRmse<<" rad/s（中心差分 + 21 点移动平均，仅用于曲线展示）\n";
    md<<"- `fit_comparison.png`：观测角度与拟合角度；`angular_velocity.png`：估计角速度与模型；`residuals.png`：角度残差；`tracking_overlay.mp4`：逐帧跟踪可视化。\n";
    md.close();

    ofstream csv(outDir+"/observations.csv");
    csv<<"frame,time_s,cx,cy,theta_rad,theta_fit_rad,residual_rad,omega_est_rad_s,omega_fit_rad_s\n";
    for(size_t k=0;k<obs.size();++k)
        csv<<obs[k].frame<<","<<obs[k].t<<","<<obs[k].x<<","<<obs[k].y<<","<<yobs[k]<<","<<yfit[k]<<","<<residual[k]<<","<<omegaObs[k]<<","<<omegaFit[k]<<"\n";

    cout<<fixed<<setprecision(8);
    cout<<"theta0="<<p(0)<<" b="<<p(1)<<" A="<<p(2)<<" Omega="<<p(3)<<" phi="<<wrapPi(p(4))<<"\n";
    cout<<"angle_RMSE="<<rmse<<" rad\n";
    return 0;
}
