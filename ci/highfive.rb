# Documentation: https://docs.brew.sh/Formula-Cookbook
#                https://rubydoc.brew.sh/Formula
# PLEASE REMOVE ALL GENERATED COMMENTS BEFORE SUBMITTING YOUR PULL REQUEST!
class Highfive < Formula
  desc "HighFive - Header-only C++ HDF5 interface"
  homepage "https://bluebrain.github.io/HighFive/"
  url "https://github.com/BlueBrain/HighFive/archive/v2.2.1.tar.gz"
  sha256 "964c722ba916259209083564405ef9ce073b15e9412955fef9281576ea9c5b85"

  depends_on "cmake" => :build
  depends_on "boost" => [ :build, :optional, :test ]
  depends_on "hdf5"

  def install
    system "cmake", ".", *std_cmake_args
    system "make"
    system "make", "install"
  end

  test do
  end
end
