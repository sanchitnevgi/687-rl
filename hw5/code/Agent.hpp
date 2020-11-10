#pragma once

#include <random>
#include <string>
#include <Eigen/Dense>

class Agent
{
public:
	// The following function should be defined as a static function in all subclasses
	// static bool updateBeforeNextAction();	// Should the agent update after getting (s,a,r,s') or after getting (s,a,r,s',a')?

	// Ask the agent to select an action given the current state
	virtual int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) = 0;

	// Tell the agent that it is at the start of a new episode
	virtual void newEpisode() = 0;

	// Depending on whether updateBeforeNextAction is true or false, one of the two below update functions must be overwritten by a subclass.
	virtual void update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator)
	{
		assert(false);	// One of the update functions must be defined. Ensure this one isn't called when it isn't defined by a subclass.
	}
	virtual void update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, const int& aPrime, std::mt19937_64& generator)
	{
		assert(false);	// One of the update functions must be defined. Ensure this one isn't called when it isn't defined by a subclass.
	}

	// Let the agent update/learn when sPrime would be the terminal absorbing state
	virtual void update(const Eigen::VectorXd& s, const int& a, const double& r, std::mt19937_64& generator) = 0;
};