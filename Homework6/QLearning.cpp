#include "QLearning.hpp"

using namespace std;
using namespace Eigen;

/*
* In this assignment, the constructor takes the environment name and
* uses it to set hyperparameters like the step size, gamma, and any other parameters.
*/
QLearning::QLearning(const int& stateDim, const int& numActions, const std::string& envName)
{
	this->numStates = stateDim;
	this->numActions = numActions;

	int basis = 1;
	env = 0;

    epsilon = 0.9;
    lambda = 0.1;

	// In the constructor, use envName to set hyperparameters.
	// No not use envName to initialize the policy!
	// I'm leaving in some of my code to show how you might structure this.
	if (envName.compare("Mountain Car") == 0)
	{
		gamma = 0.99;
		alpha = 0.005;
		basis = 2;
	}
	else if (envName.compare("Cart Pole") == 0)
	{
		gamma = 0.99;
		alpha = 0.00038;

		basis = 4;
	}
	else if (envName.compare("Gridworld") == 0)
	{
		gamma = 0.97;
		alpha = 0.4;

		env = 1;
	}
	else
	{
		cout << "Error: Unknown environment name in Sarsa constructor." << endl;
		exit(1);
	}

	if (env == 0) {
		features = ipow(basis + 1, numStates);

		c = MatrixXd::Zero(features, numStates);
		w = MatrixXd::Zero(numActions, features);
		trace = MatrixXd::Zero(numActions, features);

		VectorXd counter = VectorXd::Zero(numStates);

		for (int termCount = 0; termCount < features; termCount++) {
			c.row(termCount) = counter;
			incrementCounter(counter, basis);
		}

		phi = phiPrime = VectorXd::Zero(features);
	}
	else {
		curQ = MatrixXd::Zero(numStates, numActions);
	}
}

bool QLearning::updateBeforeNextAction()
{
	return true;
}

int QLearning::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator)
{
    VectorXd actionProbabilities;

    if (env == 1)
    {
        int state = convertOneHotToInt(s);
        actionProbabilities = curQ.row(state);
    }
    else
    {
        VectorXd features = getFeatures(s);
        actionProbabilities = (w * features);
    }

    actionProbabilities = actionProbabilities.array().exp();
    actionProbabilities.array() /= actionProbabilities.sum();

    int action;

    if (env == 1)
    {
        action = softmaxActionSelection(actionProbabilities, generator);
    }
    else
    {
        action = epsilonGreedy(actionProbabilities, generator);
    }

    return action;
}

// Tell the agent that it is at the start of a new episode
void QLearning::newEpisode()
{
	phiInit = false;
}

// Update given a (s,a,r,s') tuple
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator)
{
	if (env == 1) {
		int currState = convertOneHotToInt(s);
        int nextState = convertOneHotToInt(sPrime);

		double delta = r + gamma * curQ.row(nextState).maxCoeff() - curQ(currState, a);
		curQ(currState, a) += alpha * delta;
	}
	else
	{
		if (!phiInit) {
			phi = getFeatures(s);
			phiInit = true;
		}

		// Update trace
        for (int f = 0; f < features; ++f) {
            trace(a, f) = gamma * lambda * trace(a, f) + phi(f);
        }

		phiPrime = getFeatures(sPrime);

		double delta = r + (gamma * (w * phiPrime).maxCoeff()) - w.row(a).dot(phi);

		for (int f = 0; f < features; f++)
			w(a, f) = w(a, f) + alpha * delta * trace(a, f);

		phi = phiPrime;
	}
}

// Let the agent update/learn when sPrime would be the terminal absorbing state
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, mt19937_64& generator)
{
	if (env == 1) {
		int state = convertOneHotToInt(s);

		double delta = r - curQ(state, a);
		curQ(state, a) += alpha * delta;
	}
	else {
        for (int f = 0; f < features; ++f) {
            trace(a, f) = gamma * lambda * trace(a, f) + phi(f);
        }

		double delta = r - w.row(a).dot(phi);

		for (int i = 0; i < features; i++)
			w(a, i) = w(a, i) + alpha * delta * trace(a, i);
	}
}

int QLearning::convertOneHotToInt(VectorXd s)
{
    for (int state = 0; state < (int) s.size(); ++state)
        if (s[state] != 0)
            return state;
    return numActions - 1;
}

int QLearning::softmaxActionSelection(VectorXd probabilities, std::mt19937_64& generator)
{
    double value = uniform_real_distribution<double>(0, 1)(generator);
    double runningSum = 0;

    for (int action = 0; action < (int) probabilities.size(); ++action) {
        runningSum += probabilities[action];
        if (value <= runningSum) {
            return action;
        }
    }

    return numActions - 1;
}

int QLearning::epsilonGreedy(VectorXd probabilities, std::mt19937_64& generator)
{
    double threshold = uniform_real_distribution<double>(0, 1)(generator);

    // Select action at random
    if (threshold > epsilon)
    {
        return uniform_int_distribution<int>(0, numActions - 1)(generator);
    }
    // Select action greedily
    else
    {
        int action = 0;
        double maxProbability = 0;

        for (int a = 0; a < numActions; ++a) {
            if (probabilities[a] > maxProbability)
            {
                maxProbability = probabilities[a];
                action = a;
            }
        }

        return action;
    }
}

VectorXd QLearning::getFeatures(const VectorXd& s) {
    VectorXd result(features);

    for (int f = 0; f < features; f++) {
        result[f] = cos(M_PI * c.row(f).dot(s));
    }

    return result;
}

/*
* My code used this function. It returns a^b, where a and b are both integers.
* You may not need this, but if you do, we are including it.
*/
int QLearning::ipow(const int& a, const int& b) {
	if (b == 0) return 1;
	if (b == 1) return a;
	int tmp = ipow(a, b / 2);
	if (b % 2 == 0) return tmp * tmp;
	else return a * tmp * tmp;
}

/*
* This is another function that my code uses, and which you are welcome to use.
* It is entirely possible that your code won't require this function.
* This function takes as input a vector, buff, that represents a number in
* base (maxDigit+1), and adds one to the counter. Overflow behavior is not defined.
*/
void QLearning::incrementCounter(VectorXd& buff, const int& maxDigit) {
	for (int i = 0; i < (int)buff.size(); i++) {
		buff[i]++;
		if (buff[i] <= maxDigit)
			break;
		buff[i] = 0;
	}
}