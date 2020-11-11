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
	// @TODO: Fill in any member variables and additional functions
	int numStates;				//number of states		
	int numActions;				//number of actions
	int maxEps;					//maximum number of episodes
	int epCount;				//episode count
	double gamma;
	Eigen::MatrixXd curQ;
	double alpha;
	double epsilon;
	Eigen::MatrixXd c;
	Eigen::MatrixXd w;
	int nTerms;
	int env;

	bool phiInit = false;
	Eigen::VectorXd phi, phiPrime;

	// Two helper functions that I used. You may not need these.
	int ipow(const int& a, const int& b);
	void incrementCounter(Eigen::VectorXd& buff, const int& maxDigit);
	Eigen::VectorXd cosine_calculation(const Eigen::VectorXd& x);

};
