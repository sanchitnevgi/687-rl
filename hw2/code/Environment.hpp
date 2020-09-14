#pragma once	// This line tells the compiler: You only need to actually include this file once.
// Every time you write #include "something", the compiler effectively copy-pastes the contents of the file
// "something" where the #include statement is. If you have circular #include statements (this is allowed, and happens often!)
// then the compiler will run into issues (you will get a compilation error, which is sometimes cryptic). The statement #pragma once
// tells the compiler: If you already copied the contents of this file in previously, you can skip it this time!

#include <random>
#include <Eigen/Dense>

/*
* In C++, we create new objects called "classes". This is an example of a class.
* Each class can have functions (methods) and objects. See TabularRandomSearch.hpp for an example with objects.
* 
* Files ending in .hpp are "header" files and files ending in ".cpp" are "source" files. Generally, you define
* objects in header files, and you actually implement those definitions (the specified functions) in the source file.
* Try not to use the lines "using namespace" inside of your header files - it's bad practice. You should use those lines your source files.
* Anyway, if you create a new object, it should typically have a header file like this one, and a source file. Environment.hpp does NOT
* have a source file, Environment.cpp though! That's because Environment is an "abstract class" - it is a specification for other
* classes to use - you cannot actually make an object of type "Environment". You can make objects of type "Gridworld", which follow
* the layout described in Environment (Environment is called the "superclass" and "Gridworld" the "subclass").
* 
* All functions and objects are defined within the scope of "public:" "private:" or "protected:". 
* Things that are "public" can be seen and called at any point.
* Things that are "private" can only be referenced from within the class itself. That is, if we have a private function foo(), and we had an environment E, we could NOT call E.foo() from main(), but we could call E.foo() from within any of the functions defined inside the Environment class.
* Things that are "protected" can only be referenced from within the class or a subclass. 
* 
* This class is intended to represent an epoisodic MDP for use as depicted
* in the agent-environment diagram. This class assumes the state is a vector of reals (doubles)
*/

class Environment	// This defines the name of this object, "Environment"
{
public:

	// There are many keywords that tell the compiler something about functions and variables.
	// static: if a function is "static" it means that you don't need to actually create an object to call it. For example, transition() below depends on the current state of the environment, so it's not static. However, getNumActions() is something that we can implement without instantiating the object. So, that function is static.
	// virtual: This means that subclasses can over-write this function. If the function definition indicates "virtual" and ends in "=0", then subclasses *must* implement this function.
	//			Here we use "virtual" and "=0" to indicate that any subclass of "Environment" must specify many specific functions, and to define the form that these functions must have.
	// const: Depending on where "const" shows up, it has different meanings. Below, you'll see "const int & a". We'll talk more about that one later. Look at the "const" at the end of function definitions. This means that the function cannot make any changes to objects stored within the class.

	// The following functions should be defined as "static" functions in all subclasses.
	// static int getMaxEps();				// How many episodes should be run?
	// static int getStateDim();			// This function returns the dimension (length) of state vectors.
	// static int getNumActions();			// This function returns |\mathcal A|. Note that we are assuming that the action set is finite.
	// static double getGamma() const = 0;	// This function returns \gamma

	// This function applies action a, updating the state of the environment. It returns the reward that results from the state transition.
	virtual double transition(const int& a, std::mt19937_64 & generator) = 0;

	// This function returns the current state of the environment
	virtual Eigen::VectorXd getState() const = 0;

	// Check if the current state is Terminal Absorbing State (TAS)
	virtual bool inTAS() const = 0;

	// This function resets the environment to start a new episode (it samples the state from the initial state distribution).
	virtual void newEpisode(std::mt19937_64& generator) = 0;
};
