# Documentation: https://docs.brew.sh/Formula-Cookbook
#                https://rubydoc.brew.sh/Formula
# PLEASE REMOVE ALL GENERATED COMMENTS BEFORE SUBMITTING YOUR PULL REQUEST!
class Highfive < Formula
  desc "HighFive - Header-only C++ HDF5 interface"
  homepage "https://bluebrain.github.io/HighFive/"
  url "https://github.com/BlueBrain/HighFive/archive/v2.9.0.tar.gz"
  sha256 "6301def8ceb9f4d7a595988612db288b448a3c0546f6c83417dab38c64994d7e"

  depends_on "cmake" => :build
  depends_on "boost" => [ :build, :test ]
  depends_on "hdf5"

  def install
    system "cmake", ".", *std_cmake_args
    system "make"
    system "make", "install"
  end

  test do
  end
end
