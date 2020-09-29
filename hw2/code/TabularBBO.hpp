#pragma once

// COMPILER THAT YOU USED: [type here the compiler you used]

#include "EpisodicAgent.hpp"

/*
* You will implement this class for HW1. See TabularRandomSearch for a similar example.
*/
class TabularBBO : public EpisodicAgent
{
public:
	// Do not change the arguments provided to the constructor!
	TabularBBO(const int& stateDim, const int& numActions, const double& gamma, const int& N, const int & maxEps);

	// Ask the agent to select an action given the current state
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const override;

	// Reset the agent entirely - to a blank slate prior to learning
	void reset(std::mt19937_64& generator) override;

	void episodicUpdate(std::mt19937_64& generator) override;

    int oneHotToInt(Eigen::VectorXd s) const;

private:
    int numStates;				// How many discrete states?
    int numActions;				// How many discrete actions?
    int maxEps;					// How many episodes will be run?
    int epCount;				// Track how many episodes have been run.
    double gamma;				// Discount parameter

    Eigen::MatrixXd curTheta;	// The current best policy we have found
    double curThetaJHat;		// $\hat J(\theta_\text{cur})$ in LaTeX, this is the estimate of how good the current policy is

    Eigen::MatrixXd newTheta;	// The policy we're currently running and thinking of switching curTheta to
    double newThetaJHat;		// This will store our estimate of how good newTheta is.
};
