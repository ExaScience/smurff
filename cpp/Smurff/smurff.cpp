#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Sessions/CmdSession.h>
#include <SmurffCpp/Utils/counters.h>

int main(int argc, char** argv)
{
   using namespace smurff;

   std::shared_ptr<ISession> session = create_cmd_session(argc, argv);
   {
      COUNTER("main");
      session->run();
   }
   #ifdef PROFILING
   perf_data_print();
   #endif
   return 0;
}
