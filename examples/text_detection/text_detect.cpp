#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <caffe/util/text_detector.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " imglist_file" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];

  caffe::Detector detector(model_file, trained_file, 1);
  
  //process image one by one
  std::ifstream infile(argv[3]);
  std::string imagepath;
  while (infile >> imagepath) {
      cv::Mat img = cv::imread(imagepath);
      std::cout << img.cols << " " << img.rows << " " << img.channels() << std::endl; 
      CHECK(!img.empty()) << "Unable to decode image" << imagepath;
      std::vector<Box> dets;
      detector.Detect(img, dets);
      std::cout << "Detect " << dets.size() << " chars " << std::endl;
  }
}
