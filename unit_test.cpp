#define CATCH_CONFIG_MAIN
#include "vector_space.hpp"
#include "catch.hpp"

// test we have correct insertion
SCENARIO("VSM basic stuff", "[vsm_basics]")
{
    vector_space<std::string> vsm;
    using size_p = std::pair<size_type, size_type>;
    size_p empty_size = std::make_pair(0,0);
    REQUIRE(vsm.size() == empty_size);
    REQUIRE(vsm.is_empty() == true);

    GIVEN("a 3-word pattern") {
        pattern p_c = {"Hello", "beautiful", "world"};
        WHEN("when vectorized, must still return zero") {
            auto veh = vsm.vectorize(p_c);
            THEN("size must zero") {
                REQUIRE(veh.size() == 0);
            }
        }
    }
}

SCENARIO("VSM insertion test", "[vsm_insert]")
{
    vector_space<std::string> vsm;
    using size_p = std::pair<size_type, size_type>;

    pattern p_a = {"Hello", "world"};
    pattern p_b = {"Hello", "cruel", "world"};
    pattern p_c = {"Hello", "beautiful", "world"};

    GIVEN("insertion of a single pattern") {
        WHEN("one is inserted") {
            vsm.insert(p_a);
            THEN("VSM size must match") {
                size_p actual = std::make_pair(2,1);
                REQUIRE(vsm.size() == actual);
            }
        }
    }

    GIVEN("insertion of three patterns") {
        vsm.insert(p_a);
        vsm.insert(p_b);
        vsm.insert(p_c);

        WHEN("three are inserted") {
            THEN("VSM matrix size must match") {
                size_p actual = std::make_pair(4,3);
                REQUIRE(vsm.size() == actual);
            }
        }

        WHEN("when vectorized") {
            auto veh = vsm.vectorize(p_c);
            THEN("size must be four (count of tokens)") {
                REQUIRE(veh.size() == 4);
            }
            THEN("vector must be `1, 1, 0, 1`") {
                Eigen::VectorXf test(4);
                test << 1.f, 1.f, 0.f, 1.f;
                REQUIRE(veh == test);
            } 
        }
    }
}

SCENARIO("VSM similarity test", "[vsm_similarity]")
{
    vector_space<std::string> vsm;
    using size_p = std::pair<size_type, size_type>;

    pattern p_a = {"Hello", "world"};
    pattern p_b = {"Hello", "cruel", "world"};
    pattern p_c = {"Hello", "beautiful", "world"};

    GIVEN("insert three patterns") {
        vsm.insert(p_a);
        vsm.insert(p_b);
        vsm.insert(p_c);

        WHEN("querying `hello world`") {
            pattern query = {"Hello", "world"};
            auto sim = vsm.similar(query);
            THEN("vectors must match") {
                REQUIRE(sim == query);
            }
        }
        WHEN("querying `hello`") {
            pattern query = {"Hello"};
            auto sim = vsm.similar(query);
            //REQUIRE(min_max_sim<std::string>(sim, query) == 0.5f);
            REQUIRE(sim == p_a);
        } 
        WHEN("querying `hello beautiful girl`") {
            pattern query = {"Hello", "beautiful", "girl"};
            auto sim = vsm.similar(query);
            //REQUIRE(min_max_sim<std::string>(sim, query) == 0.666667f);
            REQUIRE(sim == p_c);
        }
        WHEN("querying `hello cruel man`") {
            pattern query = {"Hello", "cruel", "man"};
            auto sim = vsm.similar(query);
            //REQUIRE(min_max_sim<std::string>(sim, query) == 0.666667f);
            REQUIRE(sim == p_b);
        }
    }
}
