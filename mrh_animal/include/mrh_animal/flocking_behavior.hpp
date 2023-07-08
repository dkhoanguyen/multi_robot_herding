#ifndef MRH_ANIMAL__FLOCKING_BEHAVIOR_HPP_
#define MRH_ANIMAL__FLOCKING_BEHAVIOR_HPP_

#include "behavior_interface.hpp"

namespace animal
{
  class FlockingBehavior : public BehaviorInterface
  {
  public:
    FlockingBehavior();
    ~FlockingBehavior();

    bool transition();
    void update();

  protected:
  };
} // namespace animal

#endif