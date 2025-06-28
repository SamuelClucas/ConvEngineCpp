#pragma once
// Metadata container
template<typename T> 
class Label {
    public:
        const std::string filename;
        const int numObjects_;
        const std::vector<T> positions;
        Label(std::string input, int numObjs, std::vector<T> pos) 
            : filename(input),
            numObjects_(numObjs),
            positions(pos) {};
            
        ~Label() = default;

        const inline int& length() const{return this->positions.size();};
        const inline int& numObjects() const{return this->numObjects_;};
};
