#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <complex.h>

#include <Eigen/Core>

template<typename Matrix>
inline void show_internal(const char *name, const Matrix& variable)
{

   if (variable.cols()==1)
      std::cout << name << ".T (" << variable.rows() << "," << variable.cols() << ") =\n" << variable.transpose() << std::endl << std::endl;
   else
      std::cout << name << " (" << variable.rows() << "," << variable.cols() << ") =\n" << variable << std::endl << std::endl;
}


template<>
inline void show_internal(const char *name, const double& variable)
{

   std::cout << name << " =\n" << variable << std::endl << std::endl;
}

template<>
inline void show_internal(const char *name, const int& variable)
{
   std::cout << name << " =\n" << variable << std::endl << std::endl;
}

#define STRINGIFY(s) STRINGIFY_NAME(s)

#define STRINGIFY_NAME(s) #s

#define SHOW(M) show_internal(#M, M);

#define CONCAT_VAR(n1, n2) n1 ## n2

#define THROWERROR_BASE(msg, ssvar, except_type) { \
   std::stringstream ssvar; \
   ssvar << __FILE__ << ":" << __LINE__ << " in function: " << __func__ << std::endl << (msg); \
   throw except_type(ssvar.str());}

#define THROWERROR_BASE_COND(msg, ssvar, except_type, eval_cond) { \
   if(!(eval_cond)) { \
   std::stringstream ssvar; \
   ssvar << __FILE__ << ":" << __LINE__ << " in function: " << __func__ << std::endl << (msg); \
   throw except_type(ssvar.str()); }}


#define THROWERROR(msg) THROWERROR_BASE(msg, CONCAT_VAR(ss, __LINE__), std::runtime_error)

#define THROWERROR_SPEC(except_type, msg) THROWERROR_BASE(msg, CONCAT_VAR(ss, __LINE__), except_type)


#define THROWERROR_COND(msg, eval_cond) THROWERROR_BASE_COND(msg, CONCAT_VAR(ss, __LINE__), std::runtime_error, eval_cond)

#define THROWERROR_SPEC_COND(except_type, msg, eval_cond) THROWERROR_BASE_COND(msg, CONCAT_VAR(ss, __LINE__), except_type, eval_cond)


#define THROWERROR_NOTIMPL() THROWERROR_BASE(std::string("Function is not implemented:"), CONCAT_VAR(ss, __LINE__), std::runtime_error)

#define THROWERROR_NOTIMPL_MSG(msg) THROWERROR_BASE((std::string("Function is not implemented:") + msg), CONCAT_VAR(ss, __LINE__), std::runtime_error)


#define THROWERROR_FILE_NOT_EXIST(file) THROWERROR_BASE_COND((std::string("File '") + file + std::string("' not found")), CONCAT_VAR(ss, __LINE__), std::runtime_error, smurff::generic_io::file_exists(file))


#define THROWERROR_ASSERT(cond) THROWERROR_COND("assert failed: " STRINGIFY(cond), cond)

#define THROWERROR_ASSERT_MSG(cond, msg) THROWERROR_COND((std::string("assert failed: " STRINGIFY(cond) "\n")  + msg), cond)
