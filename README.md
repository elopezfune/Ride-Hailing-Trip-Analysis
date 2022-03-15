# Road_vs_Linear
Statistics on Road vs Linear distances.

A ride hailing app currently assigns new incoming trips to the closest available vehicle. To compute such distance, the app currently computes haversine distance between the pickup point and each of the available vehicles. We refer to this distance as linear

However, the expected time to reach A from B in a city is not 100% defined by Haversine distance: Cities are known to be places where huge amount of transport infrastructure (roads, highways, bridges, tunnels) is deployed to increase capacity and reduce average travel time. Interestingly, this heavy investment in infrastructure also implies that bird distance does not work so well as proxy, so the isochrones for travel time from certain location drastically differ from the perfect circle defined by bird distance, as we can see in this example from CDMX where the blue area represents that it is reachable within a 10 min drive.

In addition to this, travel times can be drastically affected by traffic, accidents, road work...So that even if a driver is only 300m away, he might need to drive for 10 min because of road work in a bridge.

The challenge is the following:
1. Should the company move towards road distance?
2. How would you improve the experimental design? Would you collect any additional data?
