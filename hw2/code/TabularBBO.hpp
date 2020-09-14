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

private:
	// @TODO: Define aditional variables here. You can also define additional functions here (or in public: if you would like)
};
