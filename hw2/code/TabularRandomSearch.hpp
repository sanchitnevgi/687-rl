#pragma once

#include "EpisodicAgent.hpp"

/*
* An extremely naive search algorithm for tabular (discrete) problems.
* It assumes that the state provided is a one-hot encoding of the discrete state.
* 
* N is the number of episodes to run before episodicUpdate will be called.
* 
* You should use this as an example for creating your own algorithm
*/
class TabularRandomSearch : public EpisodicAgent	// We are creating an object called 'TabularRandomSearch', which is a subclass of 'Episodic Agent', which is a subclass of 'Agent'
{
public:
	// This is the "constructor". It is called when the TabularRandomSearch object is created.
	// stateDim: The state will come in as a one-hot vector, and so the stateDim is actually the number of discrete states.
	// numActions: Number of actions to choose from
	// gamma: The gamma that the final policy will be evaluated using
	// N: The number of episodes that will be sampled before every call to episodicUpdate (this is already handled by the parent class).
	TabularRandomSearch(const int& stateDim, const int& numActions, const double & gamma, const int & N, const int & maxEps);

	// Ask the agent to select an action given the current state
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const override;

	// Reset the agent entirely - to a blank slate prior to learning
	void reset(std::mt19937_64& generator) override;

	// This will be called after every N episodes. See the parent class "EpisodicAgent.hpp". Notice how it works, and how it stores the data from the last N episodes in states, actions, and rewards (vectors)
	void episodicUpdate(std::mt19937_64& generator) override;

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
