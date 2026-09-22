#include "common/vision_common.hpp"
#include "task1_image/task1_config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>

int main(int argc, char** argv) {
    const std::string input = argc > 1 ? argv[1] : "resources/test_image.jpg";
    const std::string out = argc > 2 ? argv[2] : "result/task1_images/";

    cv::Mat img = cv::imread(input);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << input << "\n";
        return 1;
    }

    cv::Mat gray, hsv;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    cv::imwrite(out + "gray.png", gray);
    cv::imwrite(out + "hsv_h.png", hsv_channels[0]);
    cv::imwrite(out + "hsv_s.png", hsv_channels[1]);
    cv::imwrite(out + "hsv_v.png", hsv_channels[2]);

    cv::Mat mean_img, gaussian_img, median_img;
    cv::blur(img, mean_img, cv::Size(5, 5));
    cv::GaussianBlur(img, gaussian_img, cv::Size(5, 5), 1.5);
    cv::medianBlur(img, median_img, 5);
    cv::imwrite(out + "mean_filter.png", mean_img);
    cv::imwrite(out + "gaussian_filter.png", gaussian_img);
    cv::imwrite(out + "median_filter.png", median_img);

    cv::Mat mask_low, mask_high, red_mask;
    cv::inRange(hsv, cv::Scalar(0, 100, 100),
                cv::Scalar(10, 255, 255), mask_low);
    cv::inRange(hsv, cv::Scalar(170, 100, 100),
                cv::Scalar(179, 255, 255), mask_high);
    cv::bitwise_or(mask_low, mask_high, red_mask);
    cv::imwrite(out + "red_mask.png", red_mask);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat eroded, dilated, opened, closed;
    cv::erode(red_mask, eroded, kernel);
    cv::dilate(red_mask, dilated, kernel);
    cv::morphologyEx(red_mask, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(red_mask, closed, cv::MORPH_CLOSE, kernel);

    cv::imwrite(out + "erode.png", eroded);
    cv::imwrite(out + "dilate.png", dilated);
    cv::imwrite(out + "open.png", opened);
    cv::imwrite(out + "close.png", closed);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    cv::Mat contour_img = img.clone();
    int id = 1;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < 500.0) continue;

        const cv::Rect box = cv::boundingRect(contour);
        const double ratio =
            static_cast<double>(box.width) / box.height;
        if (ratio < 0.2 || ratio > 5.0) continue;

        cv::drawContours(contour_img,
                         std::vector<std::vector<cv::Point>>{contour},
                         0, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(contour_img, box, cv::Scalar(0, 0, 255), 2);

        std::ostringstream ss;
        ss << "ID " << id++ << " A=" << std::fixed
           << std::setprecision(0) << area;
        cv::putText(contour_img, ss.str(),
                    cv::Point(box.x, std::max(20, box.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    cv::imwrite(out + "contours_boxes.png", contour_img);

    cv::Mat drawing = img.clone();
    const int W = img.cols, H = img.rows;
    const cv::Point center(W / 2, H / 2);
    cv::circle(drawing, center, std::min(W, H) / 10,
               cv::Scalar(255, 0, 0), 4);
    cv::rectangle(drawing, cv::Rect(40, 40, W / 3 - 40, H / 3 - 40),
                  cv::Scalar(0, 255, 0), 4);
    cv::putText(drawing, "OpenCV Task 1",
                cv::Point(50, H - 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    cv::imwrite(out + "drawing.png", drawing);

    cv::Mat rotation = cv::getRotationMatrix2D(center, 35.0, 1.0);
    cv::Mat rotated;
    cv::warpAffine(img, rotated, rotation, img.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    cv::imwrite(out + "rotated_35deg.png", rotated);

    cv::Mat crop = img(cv::Rect(0, 0, W / 2, H / 2)).clone();
    cv::imwrite(out + "crop_top_left.png", crop);

    std::cout << "Task 1 completed. Results saved to " << out << "\n";
    return 0;
}
