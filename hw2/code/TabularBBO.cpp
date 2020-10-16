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
    curTheta = newTheta = MatrixXd::Zero(numStates, numActions);
    curThetaJHat = 0;

	epCount = 0;
}

// Ask the agent to select an action given the current state
int TabularBBO::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const
{
    // Convert the state from one-hot encoding to number [0, 23]
    int state = oneHotToInt(s);
    assert(state != (int)s.size());

    // Get the action logits for current state and convert to softmax probability
    VectorXd actionProbabilities = newTheta.row(state).array().exp();
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
    curTheta = newTheta = MatrixXd::Zero(numStates, numActions);
    curThetaJHat = 0;
    epCount = 0;

    EpisodicAgent::reset(generator);
}

void TabularBBO::episodicUpdate(mt19937_64& generator)
{
    // Track how many episodes have passed
    epCount += N;

    // We are going to compute newThetaJHat (an estimate of how good the new policy is), and will then
    // see if it is better than the best policy we found so far.
    newThetaJHat = 0;

    // Loop over the N episodes
    for (int epCount = 0; epCount < N; epCount++)
    {
        // Compute the return
        double curGamma = 1;
        int epLen = (int)rewards[epCount].size();
        for (int t = 0; t < epLen; t++)
        {
            newThetaJHat += curGamma * rewards[epCount][t];
            curGamma *= gamma;
        }
    }
    newThetaJHat /= (double) N;

    // Is the new policy better than our current best?
    if (newThetaJHat > curThetaJHat)
    {
        // It looks like it! Change our current best
        curTheta = newTheta;
        curThetaJHat = newThetaJHat;
    }

    // For the last 10% of episodes use the best policy
    if (epCount > (int)(maxEps*0.9))
    {
        newTheta = curTheta;
        return;
    }

    for (int s = 0; s < numStates; s++)
    {
        for (int a = 0; a < numActions; a++)
        {
            normal_distribution<double> d(curTheta(s, a), 1);
            newTheta(s, a) = d(generator);
        }
    }

}

int TabularBBO::oneHotToInt(VectorXd s) const
{
    int state;
    for (state = 0; state < (int)s.size(); state++)
        if (s[state] != 0)
            return state;

    return state - 1;
}