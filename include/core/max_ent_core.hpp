#pragma once
#include <armadillo>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h> // Add this for stdout_color_mt

class MaxEntCore
{
  public:

    int nspins;
    int nedges;

    arma::Col<double> h;
    arma::Col<double> J;
    arma::Mat<int> edges;

    std::shared_ptr<spdlog::logger> logger;


    MaxEntCore(size_t n) : nspins(n){
      nedges = nspins * (nspins - 1) / 2;
      h.set_size(nspins);
      h.fill(0);
      J.set_size(nedges);
      J.fill(0);
      edges.set_size(nspins,nspins);
      int idx = 0;
      for (int i=0; i< nspins-1; ++i){
        for (int j=i+1; j< nspins; ++j){
          edges(i,j) = edges(j,i) = idx++;
        }
      }
    
        // Initialize spdlog logger
        logger = spdlog::stdout_color_mt("core_logger");
        logger->set_level(spdlog::level::info);
        logger->info("Logger initialized in MaxEntCore");
    }


};