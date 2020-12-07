#pragma once

#define _USE_MATH_DEFINES 
#include <math.h>			// Defines M_PI
#include <iostream>

#include "Agent.hpp"

class Sarsa : public Agent
{
public:
	Sarsa(const int & stateDim, const int & numActions, const std::string & envName);	
	static bool updateBeforeNextAction();
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) override;
	void newEpisode() override;
	void update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, const int & aPrime, std::mt19937_64& generator) override;
	void update(const Eigen::VectorXd& s, const int& a, const double& r, std::mt19937_64& generator) override;

private:
    int numActions;
    int stateDim;

    // Hyper-parameters
    double alpha;
    double gamma;
    double epsilon;
    int numFeatures;

    int basis;
    bool useFourier;
    bool initPhi;

    // Weights (to be learnt) for function approximation
    Eigen::MatrixXd w;
    // Feature vector
    Eigen::VectorXd phi;
    // Q-function for GridWorld
    Eigen::MatrixXd q;

    Eigen::VectorXd getFeatures(const Eigen::VectorXd& s);
    int convertOneHotToInt(Eigen::VectorXd s);

	int softmaxActionSelection(Eigen::VectorXd probabilities, std::mt19937_64& generator);
    int epsilonGreedy(Eigen::VectorXd probabilities, std::mt19937_64& generator);

	// Two helper functions that I used. You may not need these.
	int ipow(const int& a, const int& b);
	void incrementCounter(Eigen::VectorXd& buff, const int& maxDigit);
};
