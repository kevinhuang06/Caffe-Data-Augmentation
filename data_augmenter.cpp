
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


#include "caffe/util/data_augmenter.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataAugmenter<Dtype>::DataAugmenter(const TransformationParameter& param)
    : param_(param) {
    InitRand();
    m_display_info = param_.show_augment_info();
    m_imwrite_dir  = param_.dir_to_save_augmented_imgs();
    m_img_index    = 0;
}

template <typename Dtype>
void DataAugmenter<Dtype>::InitRand() {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int DataAugmenter<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}


template <typename Dtype>
void DataAugmenter<Dtype>::Transform(cv::Mat& cv_img){

  if ( m_imwrite_dir.length() > 2){
     char im_path[256];
     sprintf(im_path, "%s/%d_ori.jpg",m_imwrite_dir.c_str(), ++m_img_index );
     cv::imwrite( im_path, cv_img);
  } 
  if ( param_.color()) { 
    Color(cv_img);
  }
  
  if ( param_.contrast()) { 
    Contrast(cv_img);
  }
  
  if ( param_.brightness()){ 
    Brightness(cv_img); 
  }
  
  if ( param_.rotation_angle_interval() > 0 ){ 
    Rotation(cv_img, param_.rotation_angle_interval()); 
  }
  if ( m_imwrite_dir.length() > 2){
     char im_path[256];
     sprintf(im_path, "%s/%d_aug.jpg",m_imwrite_dir.c_str(), m_img_index );
     cv::imwrite( im_path, cv_img);
  }
}


template <typename Dtype>
void DataAugmenter<Dtype>::Color(cv::Mat& cv_img){
  //alpha 0.8 - 1.2
  double alpha = ( Rand(5) + 8) /10.0;
  cv::Mat gray_image = cv_img.clone();
  cv::Mat degenerate;
 
  cv::cvtColor( cv_img, gray_image, CV_BGR2GRAY);
  cv::cvtColor( gray_image, degenerate, CV_GRAY2BGR);
  
  cv::addWeighted( degenerate, 1-alpha, cv_img, alpha, 0.0, cv_img);

  if (m_display_info){
    LOG(INFO) << "* Alpha for Color: " << alpha ;
  }
}
template <typename Dtype>
void DataAugmenter<Dtype>::Contrast(cv::Mat& cv_img){

  double alpha = ( Rand(5) + 8) /10.0;
  cv::Mat gray_image;
  cv::Mat degenerate;
  cv::cvtColor( cv_img, gray_image, CV_BGR2GRAY);
  gray_image.setTo(cv::mean(gray_image));
  cvtColor( gray_image, degenerate, CV_GRAY2BGR);
  cv::addWeighted( degenerate, 1-alpha, cv_img, alpha, 0.0, cv_img);

  if (m_display_info){
    LOG(INFO) << "* Alpha for Contrast: " << alpha;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Brightness(cv::Mat& cv_img){

  double alpha = ( Rand(5) + 8) /10.0;
  cv::Mat zero_img = cv_img.clone();
  zero_img.setTo(cv::Scalar(0,0,0));
  cv::addWeighted( zero_img, 1-alpha, cv_img, alpha, 0.0, cv_img);

  if (m_display_info){
    LOG(INFO) << "* Alpha for Brightness : " << alpha;
  }
}

// Rotation -------------------------------------------------------------
template <typename Dtype>
void DataAugmenter<Dtype>::Rotation(cv::Mat& cv_img ,int rotation_angle_interval){
  double rotation_degree;
  cv::Mat dst;
  int interval = 360/rotation_angle_interval;
  int apply_rotation = Rand(interval);

  cv::Size dsize = cv::Size(cv_img.cols*1.5,cv_img.rows*1.5);
  cv::Mat resize_img = cv::Mat(dsize,CV_32S);
  cv::resize(cv_img, resize_img,dsize);

  cv::Point2f pt(resize_img.cols/2., resize_img.rows/2.);    
  rotation_degree = apply_rotation*rotation_angle_interval;
  cv::Mat r = getRotationMatrix2D(pt, rotation_degree, 1.0);
  warpAffine(resize_img, dst, r, cv::Size(resize_img.cols, resize_img.rows));

  cv::Rect myROI(resize_img.cols/6, resize_img.rows/6, cv_img.cols, cv_img.rows);
  cv::Mat crop_after_rotate = dst(myROI);

  crop_after_rotate.copyTo(cv_img);

  if (m_display_info){
    LOG(INFO) << "* Degree for Rotation : " << rotation_degree;
  }
}

  INSTANTIATE_CLASS(DataAugmenter);
} //namespace caffe

