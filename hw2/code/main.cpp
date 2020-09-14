/*
* Typically main.cpp is the file containing the entry point for the program - the place it starts.
* The place it starts is a function called "main", which you'll find at the bottom of this file.
* 
* I recommend you start by reading through this file. Then read through Environment.hpp, which gives an example of how we define new objects using classes.
* From there, you can read through the other files in any order.
*/

// #include statements indicate files and libraries that should be included.
// The first ones use <>, which indicate that the compiler should already know where to
// find these. You can tell your compiler to use "additional include directories", as you'll see below,
// allowing you to write <> instead of "" (we'll see that later).

// First include built in header files - these should come with your compiler.
#include <iostream>			// This stands for Input/Output Stream. We use functions from iostream to read and write from the console
#include <fstream>			// This stands for File Stream. We use this to read and write from files
#include <random>			// This holds random number generators. Do NOT use rand(). Instead, use generator (like mt19937_64) and distributions (like uniform_int_distribution) from this library.

// Include Eigen library. This is a linear algebra library.
// This does *not* come with your compiler. I've included it in the "lib" folder.
// To get this to compiler, we need to add this "lib" folder to the list of additional include directories.
// In Visual Studio, go to Project -> Properties -> C++ -> General, and in "Additional Include Directories". Make sure you have selected "Configuration: all configurations" and "Platform: x64". Then enter the value: lib;
#include <Eigen/Dense>

// Include files that are part of this project. Using "", the compiler will search starting from the main project directory.
#include "Environment.hpp"				// This is the specification for all environments. It is called an "abstract class" because you can't create an Environment object directly, but you can create objects that are of type Environment.
#include "Gridworld.hpp"				// This is an example of an Environment - it is a subclass of the superclass Environment.
#include "Agent.hpp"					// This is the specification for all agents. It is an abstract class like Environment
#include "Manual.hpp"					// This is an agent that lets you type in actions from the console, and act as the agent.
#include "TabularRandomSearch.hpp"		// This is a very naive BBO algorithm, provided as an example. It is a subclass of EpisodicAgent, which is a subclass of Agent.
#include "TabularBBO.hpp"				// This is the agent you will create

// iostream, fstream, and random are part of the "standard libraries", std. To call functions in these libraries, you must write std:: before the call, to indicate
// that you are referring to something within std. For example, std::cout, std::mt19937_64, etc. To avoid writing std everywhere, we can just say
// "using namespace std" to indicate that the compiler should search inside of the standard libraries by default. The downside of this is that
// you can't name your own function the same thing as a function in std already (if it has the same arguments). That would be double-defining a function.
using namespace std;
using namespace Eigen;					// Just like std::, we can avoid withing Eigen:: before all of the Eigen objects. 

// Get the standard error of the vector v
// The & here is critical. When you pass something to a function, C++ makes a copy of it and passes the copy. This is slow.
// If you want to pass the actual object, and not a copy, this is much faster (and also lets the function change the object). This is
// achieved by writing & before the variable name. This is called passing v "by reference".
// Here VectorXd is an object name, like "int", and is a vector from the Eigen library. (the Xd means of any length, X)
// The "const" before VectorXd means that v cannot be changed. The form "const thing & x" is common: the & means "don't make a copy - pass the actual object, since that's faster" and the "const" says: "but don't worry, I will not change its value".
double stdError(const VectorXd& v)		// The general form is: "return_type function_name(arguments)"
{
	// The most common object types are int (integer), double (double precision floating point), bool (Boolean), VectorXd (vector), MatrixXd (matrix), and vector<type>. We'll talk about vector<type> later.
	double sampleMean = v.mean();											// First, get the mean of the vector
	double temp = 0;														// Create a floating point (double precision) equal to zero
	// Below the (int) term means "cast the next thing into an "int" type. v.size() actually returns a long integer. C++ will automatically cast it to an int to compare to i, but your compiler might give you a warning that you're comparing two different integer types. The explicit casting to an "int" here avoids that warning.
	for (int i = 0; i < (int)v.size(); i++)									// This is a basic for loop. The variable i is initialized to zero at the start, it runs as long as i < (int)v.size(), and at the end of every iteration of the loop it calls i++ (i = i + 1).
		temp += (v[i] - sampleMean) * (v[i] - sampleMean);					// temp += foo; means the same thing as temp = temp + foo;
	return sqrt(temp / (double)(v.size() - 1.0)) / sqrt((double)v.size());	// Return the standard error. The returned object must match the return type in the function delaration.
}

// Run the provided agent on the provided environment for maxEps episodes. This function returns a 
// vector containing the discounted returns (using the provided gamma) from each episode.
VectorXd runAgentEnvironment(
	Agent * agt,							// The agent to run. The * here means that this is a pointer to an agent object. "pointer" means "memory location". So, this function takes as input the location of an object in memory, and that object satisfies the spceifications of the "Agent" class.
	Environment * env,						// The environment to run on (pointer)
	const int & maxEps,						// The number of episodes to run
	const double & gamma,					// The discount factor to use
	const bool & updateBeforeNextAction,	// Does this agent update before or after the next action, A_{t+1} is chosen?
	mt19937_64 & generator)					// Random number generator to use.
{
	// Wipe the agent to start a new trial
	agt->reset(generator);	// If X is an object (instantation of some class) with a variable y, we write X.y usually. If X is a *pointer* to an object, we write X->y.
	
	// Create variables that we will use
	int curAction, newAction;
	double reward, curGamma;
	VectorXd result(maxEps), curState, newState;

	// Loop over episodes
	for (int epCount = 0; epCount < maxEps; epCount++)
	{
		// Tell the agent and environment that we're starting a new episode. For the first episode, this may be redundant if the agent and environment were just created
		env->newEpisode(generator);
		agt->newEpisode();

		// Prepare for the new episode
		result[epCount] = 0;		// We will store the return here
		curGamma = 1;				// We will store gamma^t here
		curState = env->getState();	// Get the initial state
		curAction = agt->getAction(curState, generator);	// Get the initial action

		// Loop over time
		for (int t = 0; true; t++)
		{				
			reward = env->transition(curAction, generator);	// Update the state of the environment and get the reward
			result[epCount] += curGamma * reward;			// Update the return for this episode
			curGamma *= gamma;								// Decay curGamma
			if (env->inTAS())								// Check if in the terminal absorbing state
			{
				agt->update(curState, curAction, reward, generator);	// In the terminal absorbing state, so do a special temrinal update
				break;										// Break out of the loop over time.
			}
			newState = env->getState();						// If we get here, the new state isn't the terminal absorbing state. Get the new state.
			if (updateBeforeNextAction)						// Check if we should update before computing the next action
			{
				agt->update(curState, curAction, reward, newState, generator);	// Update before getting the new action
				newAction = agt->getAction(newState, generator);				// Get the new action
			}
			else
			{
				newAction = agt->getAction(newState, generator);							// Get the new action before updating the agent
				agt->update(curState, curAction, reward, newState, newAction, generator);	// Update the agent
			}

			// Prepare for the next iteration of the t-loop, where "new" variables will be the "cur" variables.
			curAction = newAction;
			curState = newState;
		}
	}

	// Return the "result" variable, holding the returns from each episode.
	return result;
}

int main(int argc, char* argv[])
{
	// Create objects we will use
	mt19937_64 generator(0);		// Random number generator, seeded with zero
	Gridworld env(generator);		// Create the environment, in this case a Gridworld
	int numTrials = 1000;	// Specify the number of trials to run and get the number of episodes per tiral.

	// Get some values once so we don't have to keep looking them up (i.e., so we can type less later)
	int stateDim = env.getStateDim(), numActions = env.getNumActions(), maxEps = env.getMaxEps();
	double gamma = env.getGamma();
	
	/////
	// If you wnat to use the manual agent, uncomment the line below, and comment out the line defining the agent to be a TabularRandomSearch object.
	/////
	//Manual a1;					// Create the agent
	int N = 10;						// How many episodes are run between update calls within TabularRandomSearch?
	TabularRandomSearch a1(stateDim, numActions, gamma, N, maxEps);
	TabularBBO a2(stateDim, numActions, gamma, N, maxEps);

	MatrixXd returns_a1(numTrials, maxEps);	// Create a matrix to store the resulting returns. results(i,j) = the return on the j'th episode of the i'th trial.
	MatrixXd returns_a2(numTrials, maxEps);	// Same as above, but for the second agent
	cout << "Starting trial 1 of " << numTrials + 1 << endl;	// "cout" means "console out", and is our print command. Separate objects to print with the << symbol. Here we are printing a string, followed by an integer, followed by std::endl (end line).
	for (int trial = 0; trial < numTrials; trial++)	// Loop over trials
	{
		if ((trial + 1) % 100 == 0)					// % means "mod"
			cout << "Starting trial " << trial+1 << " of " << numTrials << endl;

		// Run the agent on the environment for this trial, and store the result in the trial'th row of returns.
		// The & before a1 and env indicates that we are passing pointers to a1, a2, and env. This is because runAgentEnvironment
		// won't know the type of the agent and environment, only that they meet the specifications of Agent.hpp and Environment.hpp.
		// So, on their end, these inputs are pointers to objects of unknown exact type, but which meet the Agent/Environment specifications.
		returns_a1.row(trial) = runAgentEnvironment(&a1, &env, maxEps, gamma, a1.updateBeforeNextAction(), generator).transpose();
		returns_a2.row(trial) = runAgentEnvironment(&a2, &env, maxEps, gamma, a2.updateBeforeNextAction(), generator).transpose();
	}

	// Convert returns into a vector of mean returns and the standard error (used for error bars)
	VectorXd meanReturns_a1(maxEps), stderrReturns_a1(maxEps),
		meanReturns_a2(maxEps), stderrReturns_a2(maxEps);
	for (int epCount = 0; epCount < maxEps; epCount++)
	{
		meanReturns_a1[epCount] = returns_a1.col(epCount).mean();
		stderrReturns_a1[epCount] = stdError(returns_a1.col(epCount));

		meanReturns_a2[epCount] = returns_a2.col(epCount).mean();
		stderrReturns_a2[epCount] = stdError(returns_a2.col(epCount));
	}

	// Print the results to a file
	ofstream out("out.csv");													// Create an "output file stream". This will actually create the file, but it will be empty
	out << "Tabular Random Search,BBO,TRS Error Bar,BBO Error Bar" << endl;
	for (int epCount = 0; epCount < maxEps; epCount++)
		out << meanReturns_a1[epCount] << "," << meanReturns_a2[epCount] << ','
		<< stderrReturns_a1[epCount] << "," << stderrReturns_a2[epCount] << endl;	// Just like "cout" was console out, we can write to the file stream that we created called "out".
	out.close();																// Close the file. This will happen anyway when the object "out" falls out of scope. Still, it is good practice to close your files when you are done with them.
}