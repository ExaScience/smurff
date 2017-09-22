#include "Data.h"
#include "Noiseless.h"

using namespace smurff;


int IDataDimensions::size() const
{
   return dim().dot();
}

int IDataDimensions::dim(int m) const
{
   return dim().at(m);
}

void IMeanCentering::setCenterMode(std::string c)
{
   //-- centering model
   m_center_mode = stringToCenterMode(c);
   if(m_center_mode == CenterModeTypes::CENTER_INVALID)
      throw std::runtime_error("Invalid center mode");
}

double IMeanCentering::mean(int m, int c) const
{
   assert(m_mean_computed);
   return m_mode_mean.at(m)(c);
}

void IMeanCentering::compute_mode_mean()
{
   assert(!m_mean_computed);
   m_mode_mean.resize(m_dataBase->nmode());
   for (int m = 0; m < m_dataBase->nmode(); ++m)
   {
       auto &M = m_mode_mean.at(m);
       M.resize(m_dataBase->dim(m));
       for (int n = 0; n < m_dataBase->dim(m); n++)
         M(n) = compute_mode_mean(m, n);
   }
   m_mean_computed = true;
}

int IView::nview(int mode) const
{
   return 1;
}

int IView::view(int mode, int pos) const
{
   return 0;
}

int IView::view_size(int m, int v) const
{
    return m_data_dim->dim(m);
}

INoisePrecisionMean::INoisePrecisionMean(Data* d)
{
   noise_ptr.reset(new Noiseless(d));
}

void INoisePrecisionMean::update(const SubModel& model)
{
   noise().update(model);
}

INoiseModel& INoisePrecisionMean::noise() const
{
   assert(noise_ptr);
   return *noise_ptr;
}

//===

void Data::init_post()
{
   noise().init();
}

void Data::init()
{
    init_pre();

    //compute global mean & mode-wise means
    compute_mode_mean();
    center(getCwiseMean());

    init_post();
}

std::ostream& Data::info(std::ostream& os, std::string indent)
{
   os << indent << "Type: " << name << "\n";
   os << indent << "Component-wise mean: " << getCwiseMean() << "\n";
   os << indent << "Component-wise variance: " << var_total() << "\n";
   os << indent << "Center: " << getCenterModeName() << "\n";
   os << indent << "Noise: ";
   noise().info(os, "");
   return os;
}

std::ostream& Data::status(std::ostream& os, std::string indent) const
{
   os << indent << noise().getStatus() << "\n";
   return os;
}

double Data::predict(const PVec& pos, const SubModel& model) const
{
   return model.dot(pos) + offset_to_mean(pos);
}
