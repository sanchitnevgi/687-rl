#pragma once

#include "EpisodicAgent.hpp"

/*
* You will implement this class for HW1. See TabularRandomSearch for a similar example.
*/
class REINFORCE : public EpisodicAgent
{
public:
	// Do not change the arguments provided to the constructor!
	REINFORCE(const int& stateDim, const int& numActions, const double& gamma);

	// Ask the agent to select an action given the current state
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const override;

	// Reset the agent entirely - to a blank slate prior to learning
	void reset(std::mt19937_64& generator) override;

	void episodicUpdate(std::mt19937_64& generator) override;

private:
	int numStates;				// How many discrete states?
	int numActions;				// How many discrete actions?
	double alpha;
	double gamma;				// Discount parameter
	Eigen::MatrixXd theta;		// The current best policy we have found

	int oneHotToInt(const Eigen::VectorXd& v) const;	// If v is a one-hot vector, this returns the index of the one.
};
