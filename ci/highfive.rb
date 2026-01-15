# Documentation: https://docs.brew.sh/Formula-Cookbook
#                https://rubydoc.brew.sh/Formula
# PLEASE REMOVE ALL GENERATED COMMENTS BEFORE SUBMITTING YOUR PULL REQUEST!
class Highfive < Formula
  desc "HighFive - Header-only C++ HDF5 interface"
  homepage "https://bluebrain.github.io/HighFive/"
  url "https://github.com/BlueBrain/HighFive/archive/v2.10.1.tar.gz"
  sha256 "5bfb356705c6feb9d46a0507573028b289083ec4b4607a6f36187cb916f085a7"

  depends_on "cmake" => :build
  depends_on "boost" => [ :build, :test ]
  depends_on "hdf5@1.10"

  def install
    system "cmake", ".", *std_cmake_args
    system "make"
    system "make", "install"
  end

  test do
  end
end
