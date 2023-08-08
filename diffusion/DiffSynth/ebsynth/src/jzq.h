// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#ifndef JZQ_H_
#define JZQ_H_

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <vector>
#include <string>
#include <algorithm>

#ifdef __CUDACC__
  #define JZQ_DECORATOR __host__ __device__
#else
  #define JZQ_DECORATOR
#endif

template<typename T> struct zero { static JZQ_DECORATOR T value(); };

template<typename T> JZQ_DECORATOR inline T clamp(const T& x,const T& xmin,const T& xmax);
template<typename T> JZQ_DECORATOR inline T lerp(const T& a,const T& b,float t);

inline std::string spf(const std::string fmt,...);

template<int N,typename T>
struct Vec
{
  T v[N];

  JZQ_DECORATOR Vec<N,T>();
  template<typename T2> JZQ_DECORATOR explicit Vec<N,T>(const Vec<N,T2>& u);
  explicit JZQ_DECORATOR Vec<N,T>(T v0);

  JZQ_DECORATOR Vec<N,T>(T v0,T v1);
  JZQ_DECORATOR Vec<N,T>(T v0,T v1,T v2);
  JZQ_DECORATOR Vec<N,T>(T v0,T v1,T v2,T v3);
  JZQ_DECORATOR Vec<N,T>(T v0,T v1,T v2,T v3,T v4);
  JZQ_DECORATOR Vec<N,T>(T v0,T v1,T v2,T v3,T v4,T v5);

  JZQ_DECORATOR T&       operator()(int i);
  JZQ_DECORATOR const T& operator()(int i) const;
  JZQ_DECORATOR T&       operator[](int i);
  JZQ_DECORATOR const T& operator[](int i) const;

  JZQ_DECORATOR Vec<N,T> operator*=(const Vec<N,T>& u);
  JZQ_DECORATOR Vec<N,T> operator+=(const Vec<N,T>& u);

  JZQ_DECORATOR Vec<N,T> operator*=(T s);
  JZQ_DECORATOR Vec<N,T> operator+=(T s);
};

template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator-(const Vec<N,T>& u);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator+(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator-(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator-(const Vec<N,T>& u,const T v);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator*(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator/(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator*(const T s,const Vec<N,T>& u);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator*(const Vec<N,T>& u,const T s);
template<int N,typename T> Vec<N,T> JZQ_DECORATOR operator/(const Vec<N,T>& u,const T s);

template<int N,typename T> Vec<N,bool> JZQ_DECORATOR operator<(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,bool> JZQ_DECORATOR operator>(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,bool> JZQ_DECORATOR operator<=(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,bool> JZQ_DECORATOR operator>=(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,bool> JZQ_DECORATOR operator==(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> Vec<N,bool> JZQ_DECORATOR operator!=(const Vec<N,T>& u,const Vec<N,T>& v);

template<int N,typename T> JZQ_DECORATOR inline T        dot(const Vec<N,T>& u,const Vec<N,T>& v);
template<typename T>       JZQ_DECORATOR inline T        cross(const Vec<2,T> &a,const Vec<2,T> &b);
template<typename T>       JZQ_DECORATOR inline Vec<3,T> cross(const Vec<3,T> &a,const Vec<3,T> &b);
template<int N,typename T> JZQ_DECORATOR inline T        norm(const Vec<N,T>& u);
template<int N,typename T> JZQ_DECORATOR inline Vec<N,T> normalize(const Vec<N,T>& u);
template<int N,typename T> JZQ_DECORATOR inline T        min(const Vec<N,T>& u);
template<int N,typename T> JZQ_DECORATOR inline T        max(const Vec<N,T>& u);
template<int N,typename T> JZQ_DECORATOR inline T        sum(const Vec<N,T>& u);
namespace std
{
template<int N,typename T> inline Vec<N,T> min(const Vec<N,T>& u,const Vec<N,T>& v);
template<int N,typename T> inline Vec<N,T> max(const Vec<N,T>& u,const Vec<N,T>& v);
}
template<int N,typename T> inline Vec<N,T> abs(const Vec<N,T>& x);

template<int N>            inline bool     any(const Vec<N,bool>& u);
template<int N>            inline bool     all(const Vec<N,bool>& u);

template<int M,int N,typename T>
struct Mat
{
  T m[M][N];

  Mat<M,N,T>();

  Mat<M,N,T>(T a00,T a01,
             T a10,T a11);

  Mat<M,N,T>(T a00,T a01,T a02,
             T a10,T a11,T a12,
             T a20,T a21,T a22);

  Mat<M,N,T>(T a00,T a01,T a02,T a03,
             T a10,T a11,T a12,T a13,
             T a20,T a21,T a22,T a23,
             T a30,T a31,T a32,T a33);

  T&       operator()(int i,int j);
  const T& operator()(int i,int j) const;

  T*       data();
  const T* data() const;
};

template<int M1,int N1,int M2,int N2,typename T> Mat<M1,N2,T> operator*(const Mat<M1,N1,T>& A,const Mat<M2,N2,T>& B);

template<int M,int N,typename T> Vec<M,T> operator*(const Mat<M,N,T>& A,const Vec<N,T>& u);
template<int M,int N,typename T> Vec<N,T> operator*(const Vec<M,T>& u,const Mat<M,N,T>& A);

template<int M,int N,typename T> Mat<N,M,T> transpose(const Mat<M,N,T>& A);
template<int N,typename T>       T          trace(const Mat<N,N,T>& A);
template<int N,typename T>       Mat<N,N,T> inverse(const Mat<N,N,T>& A);

template<typename T>
class Array2
{
public:
  Array2();
  Array2(int width,int height);
  explicit Array2(const Vec<2,int>& size);
  Array2(const Array2<T>& a);
  ~Array2();

  Array2&  operator=(const Array2<T>& a);

  inline T&       operator[](int i);
  inline const T& operator[](int i) const;
  inline T&       operator()(int i,int j);
  inline const T& operator()(int i,int j) const;
  inline T&       operator()(const Vec<2,int>& ij);
  inline const T& operator()(const Vec<2,int>& ij) const;

  Vec<2,int> size() const;
  int        size(int dim) const;
  int        width() const;
  int        height() const;
  int        numel() const;
  T*         data();
  const T*   data() const;
  void       clear();
  void       swap(Array2<T>& b);
  bool       empty() const;

private:
  Vec<2,int> s;
  T* d;
};

template<typename T> Vec<2,int> size(const Array2<T>& a);
template<typename T> int        size(const Array2<T>& a,int dim);
template<typename T> int        numel(const Array2<T>& a);
template<typename T> void       clear(Array2<T>* a);
template<typename T> void       swap(Array2<T>& a,Array2<T>& b);
template<typename T> T          min(const Array2<T>& a);
template<typename T> T          max(const Array2<T>& a);
template<typename T> Vec<2,T>   minmax(const Array2<T>& a);
template<typename T> Vec<2,int> argmin(const Array2<T>& a);
template<typename T> Vec<2,int> argmax(const Array2<T>& a);
template<typename T> T          sum(const Array2<T>& a);
template<typename T> void       fill(Array2<T>* a,const T& value);

template<typename T,typename F> Array2<T> apply(const Array2<T>& a,F fun);

template<typename T>
class Array3
{
public:
  Array3();
  explicit Array3(const Vec<3,int>& size);
  Array3(int width,int height,int depth);
  Array3(const Array3<T>& a);
  ~Array3();

  Array3& operator=(const Array3<T>& a);

  inline T&       operator[](int i);
  inline const T& operator[](int i) const;
  inline T&       operator()(int i,int j,int k);
  inline const T& operator()(int i,int j,int k) const;
  inline T&       operator()(const Vec<3,int>& ijk);
  inline const T& operator()(const Vec<3,int>& ijk) const;

  Vec<3,int> size() const;
  int        size(int dim) const;
  int        width() const;
  int        height() const;
  int        depth() const;
  int        numel() const;
  T*         data();
  const T*   data() const;
  void       clear();
  void       swap(Array3<T>& b);
  bool       empty() const;

private:
  Vec<3,int> s;
  T* d;
};

template<typename T> Vec<3,int> size(const Array3<T>& a);
template<typename T> int        size(const Array3<T>& a,int dim);
template<typename T> int        numel(const Array3<T>& a);
template<typename T> void       clear(Array3<T>* a);
template<typename T> void       swap(Array3<T>& a,Array3<T>& b);

typedef Vec<2,double>         Vec2d;
typedef Vec<2,float>          Vec2f;
typedef Vec<2,int>            Vec2i;
typedef Vec<2,unsigned int>   Vec2ui;
typedef Vec<2,short>          Vec2s;
typedef Vec<2,unsigned short> Vec2us;
typedef Vec<2,char>           Vec2c;
typedef Vec<2,unsigned char>  Vec2uc;

typedef Vec<3,double>         Vec3d;
typedef Vec<3,float>          Vec3f;
typedef Vec<3,int>            Vec3i;
typedef Vec<3,unsigned int>   Vec3ui;
typedef Vec<3,short>          Vec3s;
typedef Vec<3,unsigned short> Vec3us;
typedef Vec<3,char>           Vec3c;
typedef Vec<3,unsigned char>  Vec3uc;

typedef Vec<4,double>         Vec4d;
typedef Vec<4,float>          Vec4f;
typedef Vec<4,int>            Vec4i;
typedef Vec<4,unsigned int>   Vec4ui;
typedef Vec<4,short>          Vec4s;
typedef Vec<4,unsigned short> Vec4us;
typedef Vec<4,char>           Vec4c;
typedef Vec<4,unsigned char>  Vec4uc;

typedef Vec<5,double>         Vec5d;
typedef Vec<5,float>          Vec5f;
typedef Vec<5,int>            Vec5i;
typedef Vec<5,unsigned int>   Vec5ui;
typedef Vec<5,short>          Vec5s;
typedef Vec<5,unsigned short> Vec5us;
typedef Vec<5,char>           Vec5c;
typedef Vec<5,unsigned char>  Vec5uc;

typedef Vec<6,double>         Vec6d;
typedef Vec<6,float>          Vec6f;
typedef Vec<6,int>            Vec6i;
typedef Vec<6,unsigned int>   Vec6ui;
typedef Vec<6,short>          Vec6s;
typedef Vec<6,unsigned short> Vec6us;
typedef Vec<6,char>           Vec6c;
typedef Vec<6,unsigned char>  Vec6uc;

typedef Vec<2,double>         V2d;
typedef Vec<2,float>          V2f;
typedef Vec<2,int>            V2i;
typedef Vec<2,unsigned int>   V2ui;
typedef Vec<2,short>          V2s;
typedef Vec<2,unsigned short> V2us;
typedef Vec<2,char>           V2c;
typedef Vec<2,unsigned char>  V2uc;

typedef Vec<3,double>         V3d;
typedef Vec<3,float>          V3f;
typedef Vec<3,int>            V3i;
typedef Vec<3,unsigned int>   V3ui;
typedef Vec<3,short>          V3s;
typedef Vec<3,unsigned short> V3us;
typedef Vec<3,char>           V3c;
typedef Vec<3,unsigned char>  V3uc;

typedef Vec<4,double>         V4d;
typedef Vec<4,float>          V4f;
typedef Vec<4,int>            V4i;
typedef Vec<4,unsigned int>   V4ui;
typedef Vec<4,short>          V4s;
typedef Vec<4,unsigned short> V4us;
typedef Vec<4,char>           V4c;
typedef Vec<4,unsigned char>  V4uc;

typedef Vec<5,double>         V5d;
typedef Vec<5,float>          V5f;
typedef Vec<5,int>            V5i;
typedef Vec<5,unsigned int>   V5ui;
typedef Vec<5,short>          V5s;
typedef Vec<5,unsigned short> V5us;
typedef Vec<5,char>           V5c;
typedef Vec<5,unsigned char>  V5uc;

typedef Vec<6,double>         V6d;
typedef Vec<6,float>          V6f;
typedef Vec<6,int>            V6i;
typedef Vec<6,unsigned int>   V6ui;
typedef Vec<6,short>          V6s;
typedef Vec<6,unsigned short> V6us;
typedef Vec<6,char>           V6c;
typedef Vec<6,unsigned char>  V6uc;

typedef Mat<2,2,float> Mat2x2f;
typedef Mat<2,3,float> Mat2x3f;
typedef Mat<2,4,float> Mat2x4f;
typedef Mat<2,5,float> Mat2x5f;
typedef Mat<2,6,float> Mat2x6f;
typedef Mat<2,7,float> Mat2x7f;
typedef Mat<2,8,float> Mat2x8f;
typedef Mat<3,2,float> Mat3x2f;
typedef Mat<3,3,float> Mat3x3f;
typedef Mat<3,4,float> Mat3x4f;
typedef Mat<3,5,float> Mat3x5f;
typedef Mat<3,6,float> Mat3x6f;
typedef Mat<3,7,float> Mat3x7f;
typedef Mat<3,8,float> Mat3x8f;
typedef Mat<4,2,float> Mat4x2f;
typedef Mat<4,3,float> Mat4x3f;
typedef Mat<4,4,float> Mat4x4f;
typedef Mat<4,5,float> Mat4x5f;
typedef Mat<4,6,float> Mat4x6f;
typedef Mat<4,7,float> Mat4x7f;
typedef Mat<4,8,float> Mat4x8f;
typedef Mat<5,2,float> Mat5x2f;
typedef Mat<5,3,float> Mat5x3f;
typedef Mat<5,4,float> Mat5x4f;
typedef Mat<5,5,float> Mat5x5f;
typedef Mat<5,6,float> Mat5x6f;
typedef Mat<5,7,float> Mat5x7f;
typedef Mat<5,8,float> Mat5x8f;
typedef Mat<6,2,float> Mat6x2f;
typedef Mat<6,3,float> Mat6x3f;
typedef Mat<6,4,float> Mat6x4f;
typedef Mat<6,5,float> Mat6x5f;
typedef Mat<6,6,float> Mat6x6f;
typedef Mat<6,7,float> Mat6x7f;
typedef Mat<6,8,float> Mat6x8f;
typedef Mat<7,2,float> Mat7x2f;
typedef Mat<7,3,float> Mat7x3f;
typedef Mat<7,4,float> Mat7x4f;
typedef Mat<7,5,float> Mat7x5f;
typedef Mat<7,6,float> Mat7x6f;
typedef Mat<7,7,float> Mat7x7f;
typedef Mat<7,8,float> Mat7x8f;
typedef Mat<8,2,float> Mat8x2f;
typedef Mat<8,3,float> Mat8x3f;
typedef Mat<8,4,float> Mat8x4f;
typedef Mat<8,5,float> Mat8x5f;
typedef Mat<8,6,float> Mat8x6f;
typedef Mat<8,7,float> Mat8x7f;
typedef Mat<8,8,float> Mat8x8f;

typedef Mat<2,2,double> Mat2x2d;
typedef Mat<2,3,double> Mat2x3d;
typedef Mat<2,4,double> Mat2x4d;
typedef Mat<2,5,double> Mat2x5d;
typedef Mat<2,6,double> Mat2x6d;
typedef Mat<2,7,double> Mat2x7d;
typedef Mat<2,8,double> Mat2x8d;
typedef Mat<3,2,double> Mat3x2d;
typedef Mat<3,3,double> Mat3x3d;
typedef Mat<3,4,double> Mat3x4d;
typedef Mat<3,5,double> Mat3x5d;
typedef Mat<3,6,double> Mat3x6d;
typedef Mat<3,7,double> Mat3x7d;
typedef Mat<3,8,double> Mat3x8d;
typedef Mat<4,2,double> Mat4x2d;
typedef Mat<4,3,double> Mat4x3d;
typedef Mat<4,4,double> Mat4x4d;
typedef Mat<4,5,double> Mat4x5d;
typedef Mat<4,6,double> Mat4x6d;
typedef Mat<4,7,double> Mat4x7d;
typedef Mat<4,8,double> Mat4x8d;
typedef Mat<5,2,double> Mat5x2d;
typedef Mat<5,3,double> Mat5x3d;
typedef Mat<5,4,double> Mat5x4d;
typedef Mat<5,5,double> Mat5x5d;
typedef Mat<5,6,double> Mat5x6d;
typedef Mat<5,7,double> Mat5x7d;
typedef Mat<5,8,double> Mat5x8d;
typedef Mat<6,2,double> Mat6x2d;
typedef Mat<6,3,double> Mat6x3d;
typedef Mat<6,4,double> Mat6x4d;
typedef Mat<6,5,double> Mat6x5d;
typedef Mat<6,6,double> Mat6x6d;
typedef Mat<6,7,double> Mat6x7d;
typedef Mat<6,8,double> Mat6x8d;
typedef Mat<7,2,double> Mat7x2d;
typedef Mat<7,3,double> Mat7x3d;
typedef Mat<7,4,double> Mat7x4d;
typedef Mat<7,5,double> Mat7x5d;
typedef Mat<7,6,double> Mat7x6d;
typedef Mat<7,7,double> Mat7x7d;
typedef Mat<7,8,double> Mat7x8d;
typedef Mat<8,2,double> Mat8x2d;
typedef Mat<8,3,double> Mat8x3d;
typedef Mat<8,4,double> Mat8x4d;
typedef Mat<8,5,double> Mat8x5d;
typedef Mat<8,6,double> Mat8x6d;
typedef Mat<8,7,double> Mat8x7d;
typedef Mat<8,8,double> Mat8x8d;

typedef Array2<double>                  Array2d;
typedef Array2<float>                   Array2f;
typedef Array2<int>                     Array2i;
typedef Array2<unsigned int>            Array2ui;
typedef Array2<short>                   Array2s;
typedef Array2<unsigned short>          Array2us;
typedef Array2<char>                    Array2c;
typedef Array2<unsigned char>           Array2uc;

typedef Array2< Vec<2,double> >         Array2V2d;
typedef Array2< Vec<2,float> >          Array2V2f;
typedef Array2< Vec<2,int> >            Array2V2i;
typedef Array2< Vec<2,unsigned int> >   Array2V2ui;
typedef Array2< Vec<2,short> >          Array2V2s;
typedef Array2< Vec<2,unsigned short> > Array2V2us;
typedef Array2< Vec<2,char> >           Array2V2c;
typedef Array2< Vec<2,unsigned char> >  Array2V2uc;

typedef Array2< Vec<3,double> >         Array2V3d;
typedef Array2< Vec<3,float> >          Array2V3f;
typedef Array2< Vec<3,int> >            Array2V3i;
typedef Array2< Vec<3,unsigned int> >   Array2V3ui;
typedef Array2< Vec<3,short> >          Array2V3s;
typedef Array2< Vec<3,unsigned short> > Array2V3us;
typedef Array2< Vec<3,char> >           Array2V3c;
typedef Array2< Vec<3,unsigned char> >  Array2V3uc;

typedef Array2< Vec<4,double> >         Array2V4d;
typedef Array2< Vec<4,float> >          Array2V4f;
typedef Array2< Vec<4,int> >            Array2V4i;
typedef Array2< Vec<4,unsigned int> >   Array2V4ui;
typedef Array2< Vec<4,short> >          Array2V4s;
typedef Array2< Vec<4,unsigned short> > Array2V4us;
typedef Array2< Vec<4,char> >           Array2V4c;
typedef Array2< Vec<4,unsigned char> >  Array2V4uc;

typedef Array2<double>                  A2d;
typedef Array2<float>                   A2f;
typedef Array2<int>                     A2i;
typedef Array2<unsigned int>            A2ui;
typedef Array2<short>                   A2s;
typedef Array2<unsigned short>          A2us;
typedef Array2<char>                    A2c;
typedef Array2<unsigned char>           A2uc;

typedef Array2< Vec<2,double> >         A2V2d;
typedef Array2< Vec<2,float> >          A2V2f;
typedef Array2< Vec<2,int> >            A2V2i;
typedef Array2< Vec<2,unsigned int> >   A2V2ui;
typedef Array2< Vec<2,short> >          A2V2s;
typedef Array2< Vec<2,unsigned short> > A2V2us;
typedef Array2< Vec<2,char> >           A2V2c;
typedef Array2< Vec<2,unsigned char> >  A2V2uc;

typedef Array2< Vec<3,double> >         A2V3d;
typedef Array2< Vec<3,float> >          A2V3f;
typedef Array2< Vec<3,int> >            A2V3i;
typedef Array2< Vec<3,unsigned int> >   A2V3ui;
typedef Array2< Vec<3,short> >          A2V3s;
typedef Array2< Vec<3,unsigned short> > A2V3us;
typedef Array2< Vec<3,char> >           A2V3c;
typedef Array2< Vec<3,unsigned char> >  A2V3uc;

typedef Array2< Vec<4,double> >         A2V4d;
typedef Array2< Vec<4,float> >          A2V4f;
typedef Array2< Vec<4,int> >            A2V4i;
typedef Array2< Vec<4,unsigned int> >   A2V4ui;
typedef Array2< Vec<4,short> >          A2V4s;
typedef Array2< Vec<4,unsigned short> > A2V4us;
typedef Array2< Vec<4,char> >           A2V4c;
typedef Array2< Vec<4,unsigned char> >  A2V4uc;

typedef Array3<double>                  Array3d;
typedef Array3<float>                   Array3f;
typedef Array3<int>                     Array3i;
typedef Array3<unsigned int>            Array3ui;
typedef Array3<short>                   Array3s;
typedef Array3<unsigned short>          Array3us;
typedef Array3<char>                    Array3c;
typedef Array3<unsigned char>           Array3uc;

typedef Array3< Vec<2,double> >         Array3V2d;
typedef Array3< Vec<2,float> >          Array3V2f;
typedef Array3< Vec<2,int> >            Array3V2i;
typedef Array3< Vec<2,unsigned int> >   Array3V2ui;
typedef Array3< Vec<2,short> >          Array3V2s;
typedef Array3< Vec<2,unsigned short> > Array3V2us;
typedef Array3< Vec<2,char> >           Array3V2c;
typedef Array3< Vec<2,unsigned char> >  Array3V2uc;

typedef Array3< Vec<3,double> >         Array3V3d;
typedef Array3< Vec<3,float> >          Array3V3f;
typedef Array3< Vec<3,int> >            Array3V3i;
typedef Array3< Vec<3,unsigned int> >   Array3V3ui;
typedef Array3< Vec<3,short> >          Array3V3s;
typedef Array3< Vec<3,unsigned short> > Array3V3us;
typedef Array3< Vec<3,char> >           Array3V3c;
typedef Array3< Vec<3,unsigned char> >  Array3V3uc;

typedef Array3< Vec<4,double> >         Array3V4d;
typedef Array3< Vec<4,float> >          Array3V4f;
typedef Array3< Vec<4,int> >            Array3V4i;
typedef Array3< Vec<4,unsigned int> >   Array3V4ui;
typedef Array3< Vec<4,short> >          Array3V4s;
typedef Array3< Vec<4,unsigned short> > Array3V4us;
typedef Array3< Vec<4,char> >           Array3V4c;
typedef Array3< Vec<4,unsigned char> >  Array3V4uc;

typedef Array3<double>                  A3d;
typedef Array3<float>                   A3f;
typedef Array3<int>                     A3i;
typedef Array3<unsigned int>            A3ui;
typedef Array3<short>                   A3s;
typedef Array3<unsigned short>          A3us;
typedef Array3<char>                    A3c;
typedef Array3<unsigned char>           A3uc;

typedef Array3< Vec<2,double> >         A3V2d;
typedef Array3< Vec<2,float> >          A3V2f;
typedef Array3< Vec<2,int> >            A3V2i;
typedef Array3< Vec<2,unsigned int> >   A3V2ui;
typedef Array3< Vec<2,short> >          A3V2s;
typedef Array3< Vec<2,unsigned short> > A3V2us;
typedef Array3< Vec<2,char> >           A3V2c;
typedef Array3< Vec<2,unsigned char> >  A3V2uc;

typedef Array3< Vec<3,double> >         A3V3d;
typedef Array3< Vec<3,float> >          A3V3f;
typedef Array3< Vec<3,int> >            A3V3i;
typedef Array3< Vec<3,unsigned int> >   A3V3ui;
typedef Array3< Vec<3,short> >          A3V3s;
typedef Array3< Vec<3,unsigned short> > A3V3us;
typedef Array3< Vec<3,char> >           A3V3c;
typedef Array3< Vec<3,unsigned char> >  A3V3uc;

typedef Array3< Vec<4,double> >         A3V4d;
typedef Array3< Vec<4,float> >          A3V4f;
typedef Array3< Vec<4,int> >            A3V4i;
typedef Array3< Vec<4,unsigned int> >   A3V4ui;
typedef Array3< Vec<4,short> >          A3V4s;
typedef Array3< Vec<4,unsigned short> > A3V4us;
typedef Array3< Vec<4,char> >           A3V4c;
typedef Array3< Vec<4,unsigned char> >  A3V4uc;

template<> struct zero<char          > { static JZQ_DECORATOR char           value() { return 0;    } };
template<> struct zero<unsigned char > { static JZQ_DECORATOR unsigned char  value() { return 0;    } };
template<> struct zero<short         > { static JZQ_DECORATOR short          value() { return 0;    } };
template<> struct zero<unsigned short> { static JZQ_DECORATOR unsigned short value() { return 0;    } };
template<> struct zero<int           > { static JZQ_DECORATOR int            value() { return 0;    } };
template<> struct zero<unsigned int  > { static JZQ_DECORATOR unsigned int   value() { return 0;    } };
template<> struct zero<float         > { static JZQ_DECORATOR float          value() { return 0.0f; } };
template<> struct zero<double        > { static JZQ_DECORATOR double         value() { return 0.0;  } };

template<int N,typename T>
struct zero<Vec<N,T>>
{
  static JZQ_DECORATOR Vec<N,T> value()
  {
    Vec<N,T> z;
    for(int i=0;i<N;i++) { z[i] = zero<T>::value(); }
    return z;
  }
};

template<int M,int N,typename T>
struct zero<Mat<M,N,T>>
{
  static JZQ_DECORATOR Mat<M,N,T> value()
  {
    Mat<M,N,T> z;
    for(int i=0;i<M;i++)
    for(int j=0;j<N;j++)
    {
      z(i,j) = zero<T>::value();
    }
    return z;
  }
};

template <typename T> JZQ_DECORATOR inline
T clamp(const T& x,const T& xmin,const T& xmax)
{
  return std::min(std::max(x,xmin),xmax);
}

template <typename T> JZQ_DECORATOR inline
T lerp(const T& a,const T& b,float t)
{
  return (1.0f-t)*a+t*b;
}

inline std::string spf(const std::string fmt,...)
{
  int size = 1024;
  std::vector<char> buf;
  va_list ap;

  while(1)
  {
    if(size>16*1024*1024) { return std::string(""); }

    buf.resize(size);

    va_start(ap,fmt);
    const int n = vsnprintf(&buf[0],size-1,fmt.c_str(),ap);
    va_end(ap);

    if(n>-1 && n < size)
    {
      break;
    }
    else if(n>-1)
    {
      size = n + 1;
    }
    else
    {
      size = 2*size;
    }
  }

  return std::string(&buf[0]);
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T>::Vec()
{
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T>::Vec(T v0)
{
  assert(N==1);
  v[0]=v0;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T>::Vec(T v0,T v1)
{
  assert(N==2);
  v[0]=v0; v[1]=v1;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T>::Vec(T v0,T v1,T v2)
{
  assert(N==3);
  v[0]=v0; v[1]=v1; v[2]=v2;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T>::Vec(T v0,T v1,T v2,T v3)
{
  assert(N==4);
  v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T>::Vec(T v0,T v1,T v2,T v3,T v4)
{
  assert(N==5);
  v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3; v[4]=v4;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T>::Vec(T v0,T v1,T v2,T v3,T v4,T v5)
{
  assert(N==6);
  v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3; v[4]=v4; v[5]=v5;
}

template<int N,typename T> template<typename T2>
JZQ_DECORATOR
Vec<N,T>::Vec(const Vec<N,T2>& u)
{
  for(int i=0;i<N;i++)
  {
    v[i] = static_cast<T>(u.v[i]);
  }
}

template<int N,typename T>
JZQ_DECORATOR
T& Vec<N,T>::operator()(int i)
{
  assert(i>=0 && i<N);
  return v[i];
}

template<int N,typename T>
JZQ_DECORATOR
const T& Vec<N,T>::operator()(int i) const
{
  assert(i>=0 && i<N);
  return v[i];
}

template<int N,typename T>
JZQ_DECORATOR
T& Vec<N,T>::operator[](int i)
{
  assert(i>=0 && i<N);
  return v[i];
}

template<int N,typename T>
JZQ_DECORATOR
const T& Vec<N,T>::operator[](int i) const
{
  assert(i>=0 && i<N);
  return v[i];
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> Vec<N,T>::operator*=(const Vec<N,T>& u)
{
  for(int i=0;i<N;i++) v[i]*=u(i);
  return *this;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> Vec<N,T>::operator+=(const Vec<N,T>& u)
{
  for(int i=0;i<N;i++) v[i]+=u(i);
  return *this;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> Vec<N,T>::operator*=(T s)
{
  for(int i=0;i<N;i++) v[i]*=s;
  return *this;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> Vec<N,T>::operator+=(T s)
{
  for(int i=0;i<N;i++) v[i]+=s;
  return *this;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator-(const Vec<N,T>& u)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=-u(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator+(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=u(i)+v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator-(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=u(i)-v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator-(const Vec<N,T>& u,const T v)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=u(i)-v;
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator*(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=u(i)*v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator/(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=u(i)/v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator*(const T s,const Vec<N,T>& u)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=s*u(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator*(const Vec<N,T>& u,const T s)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=u(i)*s;
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,T> operator/(const Vec<N,T>& u,const T s)
{
  Vec<N,T> r;
  for(int i=0;i<N;i++) r(i)=u(i)/s;
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,bool> operator<(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,bool> r;
  for(int i=0;i<N;i++) r(i)=u(i)<v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,bool> operator>(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,bool> r;
  for(int i=0;i<N;i++) r(i)=u(i)>v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,bool> operator<=(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,bool> r;
  for(int i=0;i<N;i++) r(i)=u(i)<=v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,bool> operator>=(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,bool> r;
  for(int i=0;i<N;i++) r(i)=u(i)>=v(i);
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,bool> operator==(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,bool> r;
  for(int i=0;i<N;i++) r(i) = (u(i)==v(i));
  return r;
}

template<int N,typename T>
JZQ_DECORATOR
Vec<N,bool> operator!=(const Vec<N,T>& u,const Vec<N,T>& v)
{
  Vec<N,bool> r;
  for(int i=0;i<N;i++) r(i) = (u(i)!=v(i));
  return r;
}

template<int N,typename T>
JZQ_DECORATOR inline T dot(const Vec<N,T>& u,const Vec<N,T>& v)
{
  assert(N>0);
  T sumprod = u(0)*v(0);
  for(int i=1;i<N;i++) sumprod += u(i)*v(i);
  return sumprod;
}

template<typename T>
JZQ_DECORATOR inline T cross(const Vec<2,T> &a,const Vec<2,T> &b)
{
  return a[0]*b[1]-a[1]*b[0];
}

template<typename T>
JZQ_DECORATOR inline Vec<3,T> cross(const Vec<3,T> &a,const Vec<3,T> &b)
{
  return Vec<3,T>(a[1]*b[2]-a[2]*b[1],
                  a[2]*b[0]-a[0]*b[2],
                  a[0]*b[1]-a[1]*b[0]);
}

template<int N,typename T>
JZQ_DECORATOR inline T norm(const Vec<N,T>& u)
{
  return std::sqrt(dot(u,u));
}

template<int N,typename T>
JZQ_DECORATOR inline Vec<N,T> normalize(const Vec<N,T>& u)
{
  return u/norm(u);
}

template<int N>
JZQ_DECORATOR inline bool any(const Vec<N,bool>& u)
{
  for(int i=0;i<N;i++)
  {
    if (u(i)==true) return true;
  }
  return false;
}

template<int N>
JZQ_DECORATOR inline bool all(const Vec<N,bool>& u)
{
  for(int i=0;i<N;i++)
  {
    if (u(i)==false) return false;
  }
  return true;
}

template<int N,typename T>
JZQ_DECORATOR inline T min(const Vec<N,T>& u)
{
  assert(N>0);

  T minval = u(0);

  for(int i=1;i<N;i++)
  {
   if (u(i) < minval)
   {
      minval = u(i);
   }
  }

  return minval;
}

template<int N,typename T>
JZQ_DECORATOR inline T max(const Vec<N,T>& u)
{
  assert(N>0);

  T maxval = u(0);

  for(int i=1;i<N;i++)
  {
   if (u(i) > maxval)
   {
      maxval = u(i);
   }
  }

  return maxval;
}

template<int N,typename T>
JZQ_DECORATOR inline T sum(const Vec<N,T>& u)
{
  assert(N>0);

  T sumval = u(0);

  for(int i=1;i<N;i++)
  {
    sumval += u(i);
  }

  return sumval;
}


namespace std
{
template<int N,typename T> Vec<N,T>
inline min(const Vec<N,T>& u,const Vec<N,T>& v)
{
  assert(N>0);

  Vec<N,T> w;

  for(int i=0;i<N;i++)
  {
    w(i) = min(u(i),v(i));
  }

  return w;
}

template<int N,typename T> Vec<N,T>
inline max(const Vec<N,T>& u,const Vec<N,T>& v)
{
  assert(N>0);

  Vec<N,T> w;

  for(int i=0;i<N;i++)
  {
    w(i) = max(u(i),v(i));
  }

  return w;
}
}

template<int N,typename T> Vec<N,T>
inline abs(const Vec<N,T>& x)
{
  Vec<N,T> out;
  for(int i=0;i<N;i++) out(i) = abs(x(i));
  return out;
}

#define fori(I) for (int i=0;i<(I);i++)
#define forj(J) for (int j=0;j<(J);j++)
#define fork(K) for (int k=0;k<(K);k++)
#define forij(I,J) for (int i=0;i<(I);i++) for (int j=0;j<(J);j++)

template<int M,int N,typename T>
Mat<M,N,T>::Mat() {}

template<int M,int N,typename T>
Mat<M,N,T>::Mat(T a00,T a01,
                T a10,T a11)
{
  assert(M==2 && N==2);

  m[0][0] = a00; m[0][1] = a01;
  m[1][0] = a10; m[1][1] = a11;
}

template<int M,int N,typename T>
Mat<M,N,T>::Mat(T a00,T a01,T a02,
                T a10,T a11,T a12,
                T a20,T a21,T a22)
{
  assert(M==3 && N==3);

  m[0][0] = a00; m[0][1] = a01; m[0][2] = a02;
  m[1][0] = a10; m[1][1] = a11; m[1][2] = a12;
  m[2][0] = a20; m[2][1] = a21; m[2][2] = a22;
}

template<int M,int N,typename T>
Mat<M,N,T>::Mat(T a00,T a01,T a02,T a03,
                T a10,T a11,T a12,T a13,
                T a20,T a21,T a22,T a23,
                T a30,T a31,T a32,T a33)
{
  assert(M==4 && N==4);

  m[0][0] = a00; m[0][1] = a01; m[0][2] = a02; m[0][3] = a03;
  m[1][0] = a10; m[1][1] = a11; m[1][2] = a12; m[1][3] = a13;
  m[2][0] = a20; m[2][1] = a21; m[2][2] = a22; m[2][3] = a23;
  m[3][0] = a30; m[3][1] = a31; m[3][2] = a32; m[3][3] = a33;
}

template<int M,int N,typename T>
T& Mat<M,N,T>::operator()(int i,int j)
{
  assert(0<=i && i<M);
  assert(0<=j && j<N);
  return m[i][j];
}

template<int M,int N,typename T>
const T& Mat<M,N,T>::operator()(int i,int j) const
{
  assert(0<=i && i<M);
  assert(0<=j && j<N);
  return m[i][j];
}

template<int M,int N,typename T>
T* Mat<M,N,T>::data()
{
  return (T*)(&m[0][0]);
}

template<int M,int N,typename T>
const T* Mat<M,N,T>::data() const
{
  return (T*)(&m[0][0]);
}

template<int M1,int N1,int M2,int N2,typename T>
Mat<M1,N2,T> operator*(const Mat<M1,N1,T>& A,const Mat<M2,N2,T>& B)
{
  assert(N1==M2);
  Mat<M1,N2,T> C;

  fori(M1)
  forj(N2)
  {
    T dot = 0;
    fork(N1) dot += A(i,k) * B(k,j);
    C(i,j) = dot;
  }

  return C;
}

template<int M,int N,typename T>
Vec<M,T> operator*(const Mat<M,N,T>& A,const Vec<N,T>& u)
{
  Vec<M,T> v;

  fori(M)
  {
    T dot = 0;
    forj(N) dot += A(i,j) * u(j);
    v(i) = dot;
  }

  return v;
}

template<int M,int N,typename T>
Vec<N,T> operator*(const Vec<M,T>& u,const Mat<M,N,T>& A)
{
  Vec<N,T> v;

  forj(N)
  {
    T dot = 0;
    fori(M) dot += A(i,j) * u(i);
    v(j) = dot;
  }

  return v;
}

/*
template<int N, class T>
Mat<N,N,T> identity()
{
    Mat<N,N,T> A;
    forij(N,N) A(i,j) = ((i==j) ? 1 : 0);
    return A;
}
*/

template<int M,int N,typename T>
Mat<N,M,T> transpose(const Mat<M,N,T>& A)
{
  Mat<N,M,T> At;

  forij(N,M) At(i,j) = A(j,i);

  return At;
}

template<int N,typename T>
T trace(const Mat<N,N,T>& A)
{
  T sum = 0;

  fori(N) sum += A(i,i);

  return sum;
}

template<int N,typename T>
Mat<N,N,T> inverse(const Mat<N,N,T>& A)
{
  Mat<N,N,T> invA;

  invA = A;

  Vec<N,int> colIndex;
  Vec<N,int> rowIndex;
  Vec<N,bool> pivoted;

  fori(N) pivoted(i) = false;

  int i1, i2, row = 0, col = 0;
  T save;

  for (int i0 = 0; i0 < N; i0++)
  {
    T fMax = 0.0f;
    for (i1 = 0; i1 < N; i1++)
    {
      if (!pivoted(i1))
      {
        for (i2 = 0; i2 < N; i2++)
        {
          if (!pivoted(i2))
          {
            T fs = abs(invA(i1,i2));
            if (fs > fMax)
            {
              fMax = fs;
              row = i1;
              col = i2;
            }
          }
        }
      }
    }

    //assert(fmax > eps)

    pivoted(col) = true;

    if (row != col)
    {
        forj(N) { T tmp = invA(row,j); invA(row,j) = invA(col,j); invA(col,j) = tmp; }
    }

    rowIndex(i0) = row;
    colIndex(i0) = col;

    T inv = ((T)1.0)/invA(col,col);
    invA(col,col) = (T)1.0;
    for (i2 = 0; i2 < N; i2++)
    {
      invA(col,i2) *= inv;
    }

    for (i1 = 0; i1 < N; i1++)
    {
      if (i1 != col)
      {
        save = invA(i1,col);
        invA(i1,col) = (T)0.0;
        for (i2 = 0; i2 < N; i2++)
        {
          invA(i1,i2) -= invA(col,i2)*save;
        }
      }
    }
  }

  for (i1 = N-1; i1 >= 0; i1--)
  {
    if (rowIndex(i1) != colIndex(i1))
    {
      for (i2 = 0; i2 < N; i2++)
      {
        save = invA(i2,rowIndex(i1));
        invA(i2,rowIndex(i1)) = invA(i2,colIndex(i1));
        invA(i2,colIndex(i1)) = save;
      }
    }
  }

  return invA;
}

#undef fori
#undef forj
#undef fork

template<typename T>
Array2<T>::Array2() : s(0,0),d(0) {}

template<typename T>
Array2<T>::Array2(int width,int height)
{
  assert(width>0 && height>0);
  s = Vec2i(width,height);
  d = new T[s(0)*s(1)];
}

template<typename T>
Array2<T>::Array2(const Vec2i& size)
{
  // XXX: predelat na neco jako assert(all(s>0));
  assert(size(0)>0 && size(1)>0);
  s = size;
  d = new T[s(0)*s(1)];
}

template<typename T>
Array2<T>::Array2(const Array2<T>& a)
{
  //  printf("COPY CONSTRUCTOR\n");
  s = a.s;

  if (s(0)>0 && s(1)>0)
  {
    d = new T[s(0)*s(1)];

    // XXX: optimize this:
    for(int i=0;i<s(0)*s(1);i++) d[i] = a.d[i];
  }
  else
  {
    d = 0;
  }
}

template<typename T>
Array2<T>& Array2<T>::operator=(const Array2<T>& a)
{
  // printf("ASSIGNMENT\n");
  // printf("slow copy\n");
  if (this!=&a)
  {
    if (s(0)==a.s(0) && s(1)==a.s(1))
    {
      // XXX: optimize this:
      for(int i=0;i<s(0)*s(1);i++) d[i] = a.d[i];
      //memcpy(d,a.d,numel()*sizeof(T)); //XXX this will break down when T is not POD !!!
    }
    else
    {
      delete[] d;
      s = a.s;

      if (a.s(0)>0 && a.s(1)>0)
      {
        d = new T[s(0)*s(1)];
        //memcpy(d,a.d,numel()*sizeof(T)); //XXX this will break down when T is not POD !!!
        // XXX: optimize this:
        for(int i=0;i<s(0)*s(1);i++) d[i] = a.d[i];
      }
      else
      {
        d = 0;
      }
    }
  }
  else
  {
  //  printf("SELF ASSIGNMENT\n");
  }

  return *this;
}

template<typename T>
Array2<T>::~Array2()
{
  delete[] d;
}

template<typename T>
inline T& Array2<T>::operator[](int i)
{
  assert(i>=0 && i<numel());

  return d[i];
}

template<typename T>
inline const T& Array2<T>::operator[](int i) const
{
  assert(i>=0 && i<numel());

  return d[i];
}

template<typename T>
inline T& Array2<T>::operator()(int i,int j)
{
  assert(d!=0);
  assert(i>=0 && i<s(0) &&
         j>=0 && j<s(1));

  return d[i+j*s(0)];
}

template<typename T>
inline const T& Array2<T>::operator()(int i,int j) const
{
  assert(d!=0);
  assert(i>=0 && i<s(0) &&
         j>=0 && j<s(1));

  return d[i+j*s(0)];
}

template<typename T>
inline T& Array2<T>::operator()(const Vec<2,int>& ij)
{
  assert(d!=0);
  assert(ij(0)>=0 && ij(0)<s(0) &&
         ij(1)>=0 && ij(1)<s(1));

  return d[ij(0)+ij(1)*s(0)];
}

template<typename T>
inline const T& Array2<T>::operator()(const Vec<2,int>& ij) const
{
  assert(d!=0);
  assert(ij(0)>=0 && ij(0)<s(0) &&
         ij(1)>=0 && ij(1)<s(1));

  return d[ij(0)+ij(1)*s(0)];
}

template<typename T>
Vec2i Array2<T>::size() const
{
  return s;
}

template<typename T>
int Array2<T>::size(int dim) const
{
  assert(dim==0 || dim==1);
  return size()(dim);
}

template<typename T>
int Array2<T>::width() const
{
  return size(0);
}

template<typename T>
int Array2<T>::height() const
{
  return size(1);
}

template<typename T>
int Array2<T>::numel() const
{
  return size(0)*size(1);
}

template<typename T>
T* Array2<T>::data()
{
  return d;
}

template<typename T>
const T* Array2<T>::data() const
{
  return d;
}

template<typename T>
bool Array2<T>::empty() const
{
  return (numel()==0);
}

template<typename T>
void Array2<T>::clear()
{
  delete[] d;
  s = Vec2i(0,0);
  d = 0;
}

template<typename T>
void Array2<T>::swap(Array2<T>& b)
{
  Vec2i tmp_s = s;
  s = b.s;
  b.s = tmp_s;

  T* tmp_d = d;
  d = b.d;
  b.d = tmp_d;
}

template<typename T>
Vec2i size(const Array2<T>& a)
{
  return a.size();
}

template<typename T>
int size(const Array2<T>& a,int dim)
{
  return a.size(dim);
}

template<typename T>
int numel(const Array2<T>& a)
{
  return a.numel();
}

template<typename T>
void clear(Array2<T>* a)
{
  a->clear();
}

template<typename T>
void swap(Array2<T>& a,Array2<T>& b)
{
  a.swap(b);
}

template<typename T>
T min(const Array2<T>& a)
{
  assert(numel(a)>0);

  const int n = numel(a);

  const T* d = a.data();

  T minval = d[0];

  for(int i=1;i<n;i++) minval = (d[i]<minval) ? d[i] : minval;

  return minval;
}

template<typename T>
T max(const Array2<T>& a)
{
  assert(numel(a)>0);

  const int n = numel(a);

  const T* d = a.data();

  T maxval = d[0];

  for(int i=1;i<n;i++) maxval = (maxval<d[i]) ? d[i] : maxval;

  return maxval;
}

template<typename T>
Vec<2,T> minmax(const Array2<T>& a)
{
  assert(numel(a)>0);

  const int n = numel(a);

  const T* d = a.data();

  T minval = d[0];
  T maxval = d[0];

  for(int i=1;i<n;i++)
  {
    minval = (d[i]<minval) ? d[i] : minval;
    maxval = (maxval<d[i]) ? d[i] : maxval;
  }

  return Vec<2,T>(minval,maxval);
}

template<typename T>
Vec2i argmin(const Array2<T>& a)
{
  assert(numel(a)>0);

  T minValue = a(0,0);
  Vec2i minIndex = Vec2i(0,0);

  for(int j=0;j<a.height();j++)
  {
    for(int i=0;i<a.width();i++)
    {
      if (a(i,j)<minValue)
      {
        minValue = a(i,j);
        minIndex = Vec2i(i,j);
      }
    }
  }

  return minIndex;
}

template<typename T>
Vec2i argmax(const Array2<T>& a)
{
  assert(numel(a)>0);

  T maxValue = a(0,0);
  Vec2i maxIndex = Vec2i(0,0);

  for(int j=0;j<a.height();j++)
  {
    for(int i=0;i<a.width();i++)
    {
      if (maxValue<a(i,j))
      {
        maxValue = a(i,j);
        maxIndex = Vec2i(i,j);
      }
    }
  }

  return maxIndex;
}

template<typename T>
T sum(const Array2<T>& a)
{
  assert(numel(a)>0);

  const int n = numel(a);

  const T* d = a.data();

  T sumval = d[0];

  for(int i=1;i<n;i++) sumval += d[i];

  return sumval;
}

template<typename T>
void fill(Array2<T>* a,const T& value)
{
  assert(a!=0);
  assert(a->numel()>0);

  const int n = a->numel();
  T* d = a->data();

  for(int i=0;i<n;i++) d[i] = value;
}

template<typename T,typename F>
Array2<T> apply(const Array2<T>& a,F fun)
{
  assert(numel(a) > 0);

  Array2<T> fun_a(size(a));

  const int n = numel(a);

  for(int i=0;i<n;i++) fun_a.data()[i] = fun(a.data()[i]);

  return fun_a;
}

template<typename T>
Array3<T>::Array3() : s(0,0,0),d(0) {}

template<typename T>
Array3<T>::Array3(int width,int height,int depth)
{
  assert(width>0 && height>0 && depth>0);
  s = Vec3i(width,height,depth);
  d = new T[s(0)*s(1)*s(2)];
}

template<typename T>
Array3<T>::Array3(const Vec3i& size)
{
  // XXX: predelat na neco jako assert(all(s>0));
  assert(size(0)>0 && size(1)>0 && size(2)>0);
  s = size;
  d = new T[s(0)*s(1)*s(2)];
}

template<typename T>
Array3<T>::Array3(const Array3<T>& a)
{
  //  printf("COPY CONSTRUCTOR\n");
  s = a.s;

  if (s(0)>0 && s(1)>0 && s(2)>0)
  {
    d = new T[s(0)*s(1)*s(2)];

    // XXX: optimize this:
    for(int i=0;i<s(0)*s(1)*s(2);i++) d[i] = a.d[i];
    //memcpy((void *)d, (void *)a.d, sizeof(T)*s(0)*s(1)*s(2));
  }
  else
  {
    d = 0;
  }
}

template<typename T>
Array3<T>& Array3<T>::operator=(const Array3<T>& a)
{
  // printf("ASSIGNMENT\n");
  // printf("slow copy\n");
  if (this!=&a)
  {
    if (s(0)==a.s(0) && s(1)==a.s(1) && s(2)==a.s(2))
    {
      // XXX: optimize this:
      for(int i=0;i<s(0)*s(1)*s(2);i++) d[i] = a.d[i];
      //memcpy((void *)d, (void *)a.d, sizeof(T)*s(0)*s(1)*s(2));
    }
    else
    {
      delete[] d;
      s = a.s;

      if (a.s(0)>0 && a.s(1)>0 && a.s(2)>0)
      {
        d = new T[s(0)*s(1)*s(2)];
        // XXX: optimize this:
        for(int i=0;i<s(0)*s(1)*s(2);i++) d[i] = a.d[i];
        //memcpy((void *)d, (void *)a.d, sizeof(T)*s(0)*s(1)*s(2));
      }
      else
      {
        d = 0;
      }
    }
  }
  else
  {
  //  printf("SELF ASSIGNMENT\n");
  }

  return *this;
}

template<typename T>
Array3<T>::~Array3()
{
  delete[] d;
}

template<typename T>
inline T& Array3<T>::operator[](int i)
{
  assert(i>=0 && i<numel());

  return d[i];
}

template<typename T>
inline const T& Array3<T>::operator[](int i) const
{
  assert(i>=0 && i<numel());

  return d[i];
}

template<typename T>
inline T& Array3<T>::operator()(int i,int j,int k)
{
  assert(d!=0);
  assert(i>=0 && i<s(0) &&
         j>=0 && j<s(1) &&
         k>=0 && k<s(2));

  return d[i+(j+k*s(1))*s(0)];
}

template<typename T>
inline const T& Array3<T>::operator()(int i,int j,int k) const
{
  assert(d!=0);
  assert(i>=0 && i<s(0) &&
         j>=0 && j<s(1) &&
         k>=0 && k<s(2));

  return d[i+(j+k*s(1))*s(0)];
}

template<typename T>
inline T& Array3<T>::operator()(const Vec<3,int>& ijk)
{
  assert(d!=0);
  assert(ijk(0)>=0 && ijk(0)<s(0) &&
         ijk(1)>=0 && ijk(1)<s(1) &&
         ijk(2)>=0 && ijk(2)<s(2));

  return d[ijk(0)+(ijk(1)+ijk(2)*s(1))*s(0)];
}

template<typename T>
inline const T& Array3<T>::operator()(const Vec<3,int>& ijk) const
{
  assert(d!=0);
  assert(ijk(0)>=0 && ijk(0)<s(0) &&
         ijk(1)>=0 && ijk(1)<s(1) &&
         ijk(2)>=0 && ijk(2)<s(2));

  return d[ijk(0)+(ijk(1)+ijk(2)*s(1))*s(0)];
}

template<typename T>
Vec3i Array3<T>::size() const
{
  return s;
}

template<typename T>
int Array3<T>::size(int dim) const
{
  assert(dim==0 || dim==1 || dim==2);
  return size()(dim);
}

template<typename T>
int Array3<T>::width() const
{
  return size(0);
}

template<typename T>
int Array3<T>::height() const
{
  return size(1);
}

template<typename T>
int Array3<T>::depth() const
{
  return size(2);
}

template<typename T>
int Array3<T>::numel() const
{
  return size(0)*size(1)*size(2);
}

template<typename T>
T* Array3<T>::data()
{
  return d;
}

template<typename T>
const T* Array3<T>::data() const
{
  return d;
}

template<typename T>
void Array3<T>::clear()
{
  delete[] d;
  s = Vec3i(0,0,0);
  d = 0;
}

template<typename T>
void Array3<T>::swap(Array3<T>& b)
{
  Vec3i tmp_s = s;
  s = b.s;
  b.s = tmp_s;

  T* tmp_d = d;
  d = b.d;
  b.d = tmp_d;
}

template<typename T>
bool Array3<T>::empty() const
{
  return (numel()==0);
}

template<typename T>
Vec3i size(const Array3<T>& a)
{
  return a.size();
}

template<typename T>
int size(const Array3<T>& a,int dim)
{
  return a.size(dim);
}

template<typename T>
int numel(const Array3<T>& a)
{
  return a.numel();
}

template<typename T>
void clear(Array3<T>* a)
{
  a->clear();
}

template<typename T>
void swap(Array3<T>& a,Array3<T>& b)
{
  a.swap(b);
}

template<typename T>
void fill(Array3<T>* a,const T& value)
{
  assert(a!=0);
  assert(a->numel()>0);

  const int n = a->numel();
  T* d = a->data();

  for(int i=0;i<n;i++) d[i] = value;
}

template<typename T>
Array3<T> a3read(const std::string& fileName)
{
  Array3<T> A;
  if(!a3read(&A,fileName)) { return Array3<T>(); }
  return A;
}

template<typename T>
bool a3read(Array3<T>* out_A,const std::string& fileName)
{
  FILE* f = fopen(fileName.c_str(),"rb");

  if(!f) { return false; }

  int w,h,d,s;

  if(fread(&w,sizeof(w),1,f)!=1 ||
     fread(&h,sizeof(h),1,f)!=1 ||
     fread(&d,sizeof(d),1,f)!=1 ||
     fread(&s,sizeof(s),1,f)!=1 ||
     ((w*h*d)<1) || s!=sizeof(T))
  {
    fclose(f);
    return false;
  }

  Array3<T> A(w,h,d);

  if(fread(A.data(),sizeof(T)*w*h*d,1,f)!=1)
  {
    fclose(f);
    return false;
  }

  if(out_A!=0) { *out_A = A; }

  fclose(f);
  return true;
}

template<typename T>
bool a3write(const Array3<T>& A,const std::string& fileName)
{
  if(A.numel()==0) { return false; }

  FILE* f = fopen(fileName.c_str(),"wb");

  if(!f) { return false; }

  const int w = A.width();
  const int h = A.height();
  const int d = A.depth();
  const int s = sizeof(T);

  if(fwrite(&w,sizeof(w),1,f)!=1 ||
     fwrite(&h,sizeof(h),1,f)!=1 ||
     fwrite(&d,sizeof(d),1,f)!=1 ||
     fwrite(&s,sizeof(s),1,f)!=1 ||
     fwrite(A.data(),sizeof(T)*w*h*d,1,f)!=1)
  {
    fclose(f);
    return false;
  }

  fclose(f);
  return true;
}
#endif
