#include "TabularBBO.hpp"

using namespace std;				// So we don't have to keep writing std::
using namespace Eigen;				// So we don't have to keep writing Eigen::

// See the .hpp file for a description of this "constructor"
// Note: If you want to use a different value for N, pass that value at the end of the line below, e.g., EpisodicAgent(2)
TabularBBO::TabularBBO(const int& stateDim, const int& numActions, const double& gamma, const int& N, const int & maxEps) : EpisodicAgent(N)
{
	// @TODO: You can add code here, but do not change the argument list above.
    this->numStates = stateDim;
    this->numActions = numActions;
	this->gamma = gamma;
	this->maxEps = maxEps;
	this->N = N;

	// We begin with a uniform random policy
	theta = MatrixXd::Zero(numStates, numActions);

	thetaJHat = -INFINITY;
	epCount = 0;
}

// Ask the agent to select an action given the current state
int TabularBBO::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const
{
    // Convert the state from one-hot encoding to number [0, 23]
    int state = oneHotToInt(s);
    assert(state != (int)s.size());

    // Get the action logits for current state and convert to softmax probability
    VectorXd actionProbabilities = theta.row(state).array().exp();
    actionProbabilities.array() /= actionProbabilities.sum();

    // Sample from the probability
    double threshold = uniform_real_distribution<double>(0, 1)(generator), sum = 0;
    for (int a = 0; a < numActions; a++)
    {
        sum += actionProbabilities[a];
        if (threshold <= sum)
            return a;
    }
    return numActions - 1;
}

// Reset the agent entirely - to a blank slate prior to learning
void TabularBBO::reset(std::mt19937_64& generator)
{
    // We begin with a uniform random policy
    theta = MatrixXd::Zero(numStates, numActions);

    thetaJHat = -INFINITY;
    epCount = 0;

	EpisodicAgent::reset(generator);
}

void TabularBBO::episodicUpdate(mt19937_64& generator)
{
	// @TODO: Write this function
	epCount += N;

	double best_local_return = -INFINITY;
    MatrixXd better_theta;

    for (int ep = 0; ep < N; ++ep) {
        int epLen = (int) rewards[ep].size();

        double gamma_t = 1, local_return = 0;

        for (int t = 0; t < epLen; ++t) {
            local_return += gamma_t * rewards[ep][t];
            gamma_t *= gamma;
        }

        // Here we have the return for this episode, check if it better than best_local
        // If it is, update theta
        // Use the actions taken in the episode to create a new "policy"
        if (local_return <= best_local_return) {
            continue;
        }

        best_local_return = local_return;

        // Construct better "policy"
        better_theta = theta;

        vector<VectorXd> episode_states = states[ep];
        vector<int> episode_actions = actions[ep];

        for (int t = 0; t < epLen; ++t) {
            int given_state = oneHotToInt(episode_states[t]);
            int action_taken = episode_actions[t];

            // Since taking this action in this state, has given better reward, give it more "importance"
            better_theta.row(given_state)[action_taken] += 1;

            // TODO: Give the other actions less importance?
        }
    }

    theta = better_theta;
}

int TabularBBO::oneHotToInt(VectorXd s) const
{
    int state;
    for (state = 0; state < (int)s.size(); state++)
        if (s[state] != 0)
            return state;

    return state - 1;
}