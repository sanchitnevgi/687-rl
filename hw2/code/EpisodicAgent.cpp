#include "EpisodicAgent.hpp"

using namespace std;
using namespace Eigen;

EpisodicAgent::EpisodicAgent(const int& N)
{
	this->N = N;
	epCount = 0;
	states.resize(N);
	actions.resize(N);
	rewards.resize(N);
}

bool EpisodicAgent::updateBeforeNextAction()
{
	return true;
}

// Tell the agent that it is at the start of a new episode
void EpisodicAgent::newEpisode()
{
}

// Reset the agent entirely - to a blank slate prior to learning
void EpisodicAgent::reset(std::mt19937_64& generator)
{
	epCount = 0;
	wipeStatesActionsRewards();
}

// Update given a (s,a,r,s') tuple
void EpisodicAgent::update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator)
{
	states[epCount].push_back(s);
	actions[epCount].push_back(a);
	rewards[epCount].push_back(r);
}

// Let the agent update/learn when sPrime would be the terminal absorbing state
void EpisodicAgent::update(const Eigen::VectorXd& s, const int& a, const double& r, mt19937_64 & generator)
{
	states[epCount].push_back(s);
	actions[epCount].push_back(a);
	rewards[epCount].push_back(r);

	// Increment episode counter
	epCount++;

	if (epCount == N)					// If ready to update, update and wipe the states, actions, and rewards
	{
		episodicUpdate(generator);
		wipeStatesActionsRewards();
		epCount = 0;
	}
}

void EpisodicAgent::wipeStatesActionsRewards()
{
	for (int i = 0; i < N; i++)
	{
		states[i].resize(0);
		actions[i].resize(0);
		rewards[i].resize(0);
	}
}