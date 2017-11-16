#ifndef TEXT_UNICORN
#define TEXT_UNICORN
/*
 @brief Text Unicorn
 @date November 3rd 2017
 @author Alex Giokas
 @version 0.1.0


                             \
                              \
                               \\
                                \\
                                 >\/7
                             _.-(6'  \
                            (=___._/` \
                                 )  \ |
                                /   / |
                               /    > /
                              j    < _\
                          _.-' :      ``.
                          \ r=._\        `.
                         <`\\_  \         .`-.
                          \ r-7  `-. ._  ' .  `\
                           \`,      `-.`7  7)   )
                            \/         \|  \'  / `-._
                                       ||    .'
                                        \\  (
                                         >\  >
                                     ,.-' >.'
                                    <.'_.''
                                      <'
 */
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
///
/// a pattern is a sentence tokenized
/// we default it to a vector of strings
///
using pattern = std::vector<std::string>;
using size_type = pattern::size_type;

/// @return min-max similarity for two `pattern_type` params
template <class token_type,
          class pattern_type = std::vector<token_type>>
float min_max_sim(const pattern_type & lhs,
                  const pattern_type & rhs);
/**
 * @class vector_space
 * @brief a vector space model matrix
 * @note the template patameter `token_type` must be
 *       "searchable" inside containers for this class to
 *       function properly.
 * @note the default `pattern_type` template parameter
 *       can be changed to any stl container,
 *       provided that searching and indexing works OK
 */
template <class token_type,
          class pattern_type = std::vector<token_type>>
class vector_space
{
public:
    /// @brief insert a new pattern into the VSM
    void insert(pattern_type arg);

    /** 
     * @return similarity coefficients of @param arg
     * which are filtered using @param theta
     * @note  `M * V / ||M|| * ||V||` 
     *        where `M` is the matrix
     *        and `I` is the input vector
     */
    pattern_type similar(pattern_type arg) const;

    /// @brief print matrix on screen
    void print() const;

    /// @brief save all data
    void save(std::string file) const;

    /// @brief return size of VSM: columns * rows
    std::pair<size_type, size_type> size() const;

    /// @return is empty when both columns are zero
    bool is_empty() const;

    /// @return index of @param item inside @param container
    template <class item_type,
              class list_type = std::vector<item_type>>
    int has_index(item_type item,
                  list_type & container) const;

    /// @return the vectorized representation of pattern_type
    Eigen::VectorXf vectorize(pattern_type arg) const;

	// TODO: @brief nuke this VSM
	// void clear();

protected:
    /// TODO: @brief setup the weights matrix from the boolean matrix
    void set_weights_matrix();

private:
    // Binary Matrix
    Eigen::MatrixXf matrix;
    // Weighted (tf-idf) Matrix
    Eigen::MatrixXf weight;
    // column index (tokens), row index (patterns)
    std::vector<token_type>    col_index;
    std::vector<pattern_type>  row_index;
};
/**************************************************************
 *
 *              Template Implementations Yo!
 *
 * ***********************************************************/
template <class token_type,
          class pattern_type>
float min_max_sim(const pattern_type & lhs,
                  const pattern_type & rhs)
{
    pattern_type mine(lhs);
    pattern_type other(rhs);
    pattern_type same;
    std::sort(mine.begin(), mine.end());
    std::sort(other.begin(), other.end());
    std::set_intersection(mine.begin(), mine.end(),
                          other.begin(), other.end(),
                          std::back_inserter(same));
    float x = same.size();
    float max = std::max(mine.size(), other.size());
    return ((x - 0.f) / (max - 0.f));
}

template <class token_type,
          class pattern_type>
void vector_space<token_type,
                  pattern_type>::insert(pattern_type arg)
{
    if (has_index(arg, row_index) >= 0) {
        return;
    }
    row_index.push_back(arg);
    auto i = row_index.size() - 1;
    matrix.conservativeResize(matrix.rows() + 1, Eigen::NoChange_t());
    matrix.row(i).setZero();
    for (const auto & token : arg) {
        int k = has_index(token, col_index);
        if (k < 0) {
            matrix.conservativeResize(Eigen::NoChange_t(), matrix.cols() + 1);
            col_index.push_back(token);
            k = col_index.size() - 1;
            matrix.col(k).setZero();
        }
        matrix(i,k) = 1;
    }
}

template <class token_type,
          class pattern_type>
void vector_space<token_type,
                  pattern_type>::set_weights_matrix()
{
   // TODO: run ONCE?
   // TODO: iterate all col_index (patterns) #1
   //       find occurance of each token (number) and TF'it using #num / #total tokens
   //           TF = #num(token) / #num(pattern tokens)
   //
   //       find for count (col_index) = (patterns) with this Token
   //           IDF = log_e(#Patterns / #Patterns with Token)
   //
   //       final weight is = tf * idf
}

template <class token_type,
          class pattern_type>
void vector_space<token_type,
                  pattern_type>::print() const
{
    std::cout << matrix << std::endl;  
}

template <class token_type,
          class pattern_type>
void vector_space<token_type,
                  pattern_type>::save(std::string output) const
{
    std::string matrix_name = output;
    std::ofstream file(matrix_name);
    file << matrix << std::endl;
    file.close();
}

template <class token_type,
          class pattern_type>
template <class item_type,
          class list_type>
int vector_space<token_type,
                 pattern_type>::has_index(item_type item,
                                          list_type & container) const
{
    auto it = std::find(container.begin(), container.end(), item);
    if (it == container.end()) {
        return -1;
    }
    return std::distance(container.begin(), it); 
}

template <class token_type,
          class pattern_type>
std::pair<size_type, size_type>
    vector_space<token_type,pattern_type>::size() const
{
    return std::make_pair(col_index.size(), row_index.size());
}

template <class token_type,
          class pattern_type>
bool vector_space<token_type,
                  pattern_type>::is_empty() const
{
    return size().first == 0 && size().second == 0;
}

template <class token_type,
          class pattern_type>
pattern_type vector_space<token_type,
                          pattern_type>::similar(pattern_type arg) const
{
    std::vector<pattern_type> max_vec;
    Eigen::VectorXf veh = vectorize(arg);
	auto den = matrix.norm() * veh.norm();
	auto res = (matrix * veh) / den;
    auto max = res.maxCoeff();
    for (int i = 0; i < res.size(); ++i) {
        if (res[i] == max) {
            max_vec.push_back(row_index.at(i));
        }
    }
    if (!max_vec.empty()) {
        auto min = *std::min_element(max_vec.begin(), max_vec.end(), 
                [](const pattern_type& a, const pattern_type& b) {
                    return a.size() < b.size();
        });
        return min;
    }
    std::vector<std::string> empty{};
    return empty;
}

template <class token_type,
          class pattern_type>
Eigen::VectorXf vector_space<token_type,
                             pattern_type>::vectorize(pattern_type arg) const
{
    auto veh = Eigen::VectorXf(matrix.cols());
    veh.setZero();
    for (const auto & item : arg) {
        auto i = has_index(item, col_index);
        if (i >= 0) {
            veh(i) = 1.f;
        }
    }
    return veh;
}
#endif
