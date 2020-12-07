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
    int numStates;
    int numActions;
    int env;

    // Hyper-parameters
    double gamma;
    double alpha;
    double epsilon;
    int features;
    bool phiInit = false;
    double lambda;

    Eigen::MatrixXd c;
    Eigen::MatrixXd w;
    Eigen::MatrixXd trace;
    Eigen::MatrixXd curQ;
    Eigen::VectorXd phi, phiPrime;

    int softmaxActionSelection(Eigen::VectorXd probabilities, std::mt19937_64& generator);
    int convertOneHotToInt(Eigen::VectorXd s);
    int epsilonGreedy(Eigen::VectorXd probabilities, std::mt19937_64& generator);
    Eigen::VectorXd getFeatures(const Eigen::VectorXd& x);

	// Two helper functions that I used. You may not need these.
	int ipow(const int& a, const int& b);
	void incrementCounter(Eigen::VectorXd& buff, const int& maxDigit);
};
