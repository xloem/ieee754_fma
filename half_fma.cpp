#include <half/half.hpp>
#include <time.h>
#include <stdlib.h>
#include <cassert>
#include <type_traits>

using namespace half_float;

float half2float(half const & h)
{
    return float(h);
}

template <typename T, typename V> T & bits_cast(V & value)
{
    return *(T*)&value;
}
template <typename T, typename V> constexpr T bits_cast(V const & value)
{
    union {
        V v;
        T t;
    } unioned = {.v = value};
    return unioned.t;
}

// simplifies further if a construct is used to look up types by bitwidth they'll contain.
    // [don't know offhand how to quickly make e.g. a template that responds to a set of values within an integer range]
template <typename Float_t, typename UInt_t, typename ULongInt_t, int BITS_EXP_i>
struct IEEE754_data {
	using Float = Float_t;
	using UInt = typename std::make_unsigned<UInt_t>::type;
    using Int = typename std::make_signed<UInt_t>::type;
    using ULongInt = ULongInt_t;

    static constexpr unsigned int BITS = sizeof(Float)*8;
    static constexpr unsigned int BITS_SIGN = 1;
    static constexpr unsigned int BITS_EXP = BITS_EXP_i;
    static constexpr unsigned int BITS_M = BITS - BITS_SIGN - BITS_EXP;

    static constexpr unsigned int POS_MOSTSIG = BITS - 1;
    static constexpr unsigned int POS_LEASTSIG = 0;
    static constexpr unsigned int POS_SIGN = POS_MOSTSIG;
    static constexpr unsigned int POS_EXP = BITS_M;
    static constexpr unsigned int POS_EXP_MOSTSIG = POS_SIGN - 1;
    static constexpr unsigned int POS_EXP_LEASTSIG = POS_EXP;
    static constexpr unsigned int POS_M_MOSTSIG = POS_EXP - 1;
    static constexpr unsigned int POS_M_LEASTSIG = 0;

    static constexpr UInt BIT_LEASTSIG = UInt{1};
    static constexpr UInt BIT_MOSTSIG = BIT_LEASTSIG << POS_MOSTSIG;
    static constexpr UInt BIT_SIGN = BIT_LEASTSIG << POS_SIGN;
    static constexpr UInt MASK_ABS = static_cast<UInt>(~BIT_SIGN);
    static constexpr UInt MASK_EXP = MASK_ABS - ((BIT_LEASTSIG<<BITS_M)-1);
    static constexpr UInt BIT_EXP_MOSTSIG = (BIT_LEASTSIG << POS_EXP_MOSTSIG);
    static constexpr UInt BIT_EXP_LEASTSIG = (BIT_LEASTSIG << POS_EXP_LEASTSIG);
    static constexpr UInt MASK_M = (BIT_LEASTSIG<<BITS_M) - 1;
    static constexpr UInt BIT_M_MOSTSIG = (BIT_LEASTSIG << POS_M_MOSTSIG);
    static constexpr UInt BIT_M_LEASTSIG = (BIT_LEASTSIG << POS_M_LEASTSIG);

	static constexpr UInt EXP_BIAS = (BIT_LEASTSIG<<BITS_EXP) / 2;
    static constexpr UInt EXP_MAX = (BIT_LEASTSIG<<BITS_EXP) - 1;
	static constexpr UInt INF = MASK_EXP;
    static constexpr UInt MAX = INF - 1;
    static constexpr UInt NAN_ = INF;
    static constexpr UInt NAN_QUIET = MASK_ABS;
};

using HALF = IEEE754_data<uint16_t, uint16_t, uint32_t, 5>;
using SINGLE = IEEE754_data<float, uint32_t, uint64_t, 8>;
struct uint128_t {
    uint64_t hi, lo;
};
using DOUBLE = IEEE754_data<double, uint64_t, uint128_t, 11>;

template <class Float, std::float_round_style R = std::round_to_nearest>
typename Float::UInt inline float_round(typename Float::UInt const & value, bool guard, bool sticky)
{
	return	(R==std::round_to_nearest) ? (value+(guard&(sticky|value))) :
			(R==std::round_toward_infinity) ? (value+(~(value>>(Float::BITS-1))&(guard|sticky))) :
			(R==std::round_toward_neg_infinity) ? (value+((value>>(Float::BITS-1))&(guard|sticky))) :
			value;
}

template <class Float1, class Float=HALF, std::float_round_style R = std::round_to_nearest>
typename Float::UInt float_downcast(typename Float1::UInt bits1)
{
    constexpr auto const BITS_DIFF = Float1::BITS - Float::BITS;
	constexpr auto const BITS_M_DIFF = Float1::BITS_M - Float::BITS_M;
	constexpr auto const EXP_BIAS_DIFF = Float1::EXP_BIAS - Float::EXP_BIAS;
	//UInt1 bits1 = bits_cast<UInt1>(d);
    typename Float::UInt sign = static_cast<typename Float::UInt>(bits1>>BITS_DIFF) & Float::BIT_SIGN;
	bits1 &= Float1::MASK_ABS;
	if (bits1 > Float1::INF)
		// nan
		return sign | Float::INF | Float::BIT_M_MOSTSIG | ((bits1>>BITS_M_DIFF)&Float::MASK_M);
	else if (bits1 == Float1::INF)
		// inf
		return sign | Float::INF;
	else if (bits1 >= (Float1::EXP_BIAS + Float::EXP_BIAS - 1) << Float1::BITS_M)
		// overflow -> infinity
            // it's notable here that summation is used instead of condition, for handling positive and negative extrema together
		return	(R==std::round_toward_infinity) ? (sign+Float::INF-(sign>>Float::POS_SIGN)) :
				(R==std::round_toward_neg_infinity) ? (sign+Float::INF-1+(sign>>Float::POS_SIGN)) :
				(R==std::round_toward_zero) ? (sign|(Float::INF-1)) :
				(sign|Float::INF);
	else if (bits1 >= (Float1::EXP_BIAS - Float::EXP_BIAS + 1) << Float1::POS_EXP)
		// normal value
		return float_round<Float,R>(
			sign|(((bits1>>Float1::POS_EXP)-EXP_BIAS_DIFF)<<Float::POS_EXP)|((bits1>>BITS_M_DIFF)&Float::MASK_M), 
			(bits1>>(BITS_M_DIFF-1))&1,
			(bits1&(Float1::MASK_M>>(Float::BITS_M+1)))!=0
		);
	else if (bits1 >= (Float1::EXP_BIAS - Float::EXP_BIAS - Float::BITS_M) << Float1::BITS_M) {
		// subnormal value
		int i = Float1::EXP_BIAS - 3 - (bits1 >> Float1::BITS_M);
		bits1 = (bits1&Float1::MASK_M) | Float1::BIT_EXP_LEASTSIG;
		return float_round<Float,R>(
			sign | (bits1>>(i+1)),
			(bits1>>i)&1,
			(bits1&((typename Float::UInt{1}<<i)-1))!=0
		);
	}
	else if (bits1 != 0)
		// underflow
		return	(R==std::round_toward_infinity) ? (sign+1-(sign>>(Float::BITS-1))) :
				(R==std::round_toward_neg_infinity) ? (sign+(sign>>(Float::BITS-1))) :
				sign;
	else
		// signed zero
		return sign;
}

template <typename Float, std::float_round_style R = std::round_to_nearest>
typename Float::UInt float_fma(typename Float::UInt x, typename Float::UInt y, typename Float::UInt z)
{
		typename Float::UInt absx = x & Float::MASK_ABS, absy = y & Float::MASK_ABS, absz = z & Float::MASK_ABS;
        typename Float::Int exp = -Float::EXP_BIAS+1;
		typename Float::UInt sign = (x^y) & Float::BIT_SIGN;
		bool sub = ((sign^z)&Float::BIT_SIGN) != 0;
		if(absx >= Float::INF || absy >= Float::INF || absz >= Float::INF)
			return	//(absx>0x7C00 || absy>0x7C00 || absz>0x7C00) ? half(detail::binary, detail::signal(x, y, z)) :
                    (absx>Float::INF) ? (x|Float::BIT_M_MOSTSIG) :
                    (absy>Float::INF) ? (y|Float::BIT_M_MOSTSIG) :
                    (absz>Float::INF) ? (z|Float::BIT_M_MOSTSIG) :
					(absx==Float::INF) ? /*half(detail::binary, */(!absy || (sub && absz==Float::INF)) ? Float::NAN_QUIET/*detail::invalid()*/ : (sign|Float::INF)/*)*/ :
					(absy==Float::INF) ? /*half(detail::binary, */(!absx || (sub && absz==Float::INF)) ? Float::NAN_QUIET/*detail::invalid()*/ : (sign|Float::INF)/*)*/ : z;
		if(!absx || !absy)
			return absz ? z : /*half(detail::binary, (half::round_style*/(R==std::round_toward_neg_infinity) ? (z|sign) : (z&sign)/*)*/;
		for(; absx<Float::BIT_EXP_LEASTSIG; absx<<=1,--exp) ;
		for(; absy<Float::BIT_EXP_LEASTSIG; absy<<=1,--exp) ;
		typename Float::ULongInt m =
            static_cast<typename Float::ULongInt>(
                (absx&Float::MASK_M)|Float::BIT_EXP_LEASTSIG
            ) * static_cast<typename Float::ULongInt>(
                (absy&Float::MASK_M)|Float::BIT_EXP_LEASTSIG
            );
		int i = m >> (Float::BITS_M*2+1);
		exp += (absx>>Float::BITS_M) + (absy>>Float::BITS_M) + i;
		m <<= 3 - i;
		if(absz)
		{
			typename Float::Int expz = 0;
			for(; absz<Float::BIT_EXP_LEASTSIG; absz<<=1,--expz) ;
			expz += absz >> Float::BITS_M;
			typename Float::ULongInt mz = static_cast<typename Float::ULongInt>((absz&Float::MASK_M)|Float::BIT_EXP_LEASTSIG) << (Float::BITS_M+3);
			if(expz > exp || (expz == exp && mz > m))
			{
				std::swap(m, mz);
				std::swap(exp, expz);
				if(sub)
					sign = z & Float::BIT_SIGN;
			}
			int d = exp - expz;
			// m and mz are both 24-bit
			// the purpose of the |(..) is to ensure that half values with a little extra round upward.
			// the effect may be slightly different than correct
			mz = (d<(Float::BITS_M*2+3)) ? ((mz>>d)|((mz&((typename Float::ULongInt{1}<<d)-1))!=0)) : 1;
			if(sub)
			{
				m = m - mz;
				if(!m)
					return /*half(detail::binary, */typename Float::UInt{/*half::round_style*/R==std::round_toward_neg_infinity}<<Float::POS_SIGN/*)*/;
				for(; m<Float::BIT_EXP_LEASTSIG; m<<=1,--exp) ;
			}
			else
			{
				m += mz;
				i = m >> (Float::BITS_M*2+4);
				// this looks informative
				// if i == 1,
				// then m = m/2 | m&1
				// note: m/2 can be odd or even
				// |m&1 then means, if the next bit is set,
				// ensure the preceding bit is set
				// 3.5 -> 3 ?
				// 2.5 -> 3 ?
				// can i be >1 ?
				m = (m>>i) | (m&i);
				exp += i;
			}
		}
		if(exp >= Float::EXP_MAX)
			//return half(detail::binary, detail::overflow<half::round_style>(sign));
			return	(R==std::round_toward_infinity) ? (sign+Float::INF-(sign>>Float::POS_SIGN)) :
					(R==std::round_toward_neg_infinity) ? (sign+Float::INF-1+(sign>>Float::POS_SIGN)) :
					(R==std::round_toward_zero) ? (sign|(Float::INF-1)) :
					(sign|Float::INF);
		else if(exp < -Float::BITS_M)
		//	return half(detail::binary, detail::underflow<half::round_style>(sign));
			return	(R==std::round_toward_infinity) ? (sign+1-(sign>>Float::POS_SIGN)) :
					(R==std::round_toward_neg_infinity) ? (sign+(sign>>Float::POS_SIGN)) :
					sign;
		//return half(detail::binary, detail::fixed2half<half::round_style,23,false,false,false>(m, exp-1, sign));
        //std::float_round_style R = half::round_style;
        unsigned int F = Float::BITS_M*2+3;
        //bool S = false;
        //bool N = false;
        //bool I = false;
        exp -= 1;
        int s = 0;
			/*if(S)
			{
				uint32 msign = sign_mask(m);
				m = (m^msign) - msign;
				sign = msign & 0x8000;
			}*/
            unsigned int value;
            int guard_bit;
            int sticky_bit;
			/*if(N)
				for(; m<(static_cast<uint32>(1)<<F) && exp; m<<=1,--exp) ;
			else*/ if(exp < 0) {
                value = 
				/*return rounded<R,I>(*/sign+(m>>(F-Float::BITS_M-exp)); guard_bit= (m>>(F-Float::BITS_M-1-exp))&1; sticky_bit = s|((m&((Float::BIT_LEASTSIG<<(F-Float::BITS_M-1-exp))-1))!=0);
            } else {
                value =
    			/*return rounded<R,I>(*/sign+(exp<<Float::BITS_M)+(m>>(F-Float::BITS_M)); guard_bit= (m>>(F-Float::BITS_M-1))&1; sticky_bit = s|((m&((Float::BIT_LEASTSIG<<(F-Float::BITS_M-1))-1))!=0);
            }
			return	(R==std::round_to_nearest) ? (value+(guard_bit&(sticky_bit|value))) :
					(R==std::round_toward_infinity) ? (value+(~(value>>Float::POS_SIGN)&(guard_bit|sticky_bit))) :
					(R==std::round_toward_neg_infinity) ? (value+((value>>Float::POS_SIGN)&(guard_bit|sticky_bit))) :
					value;
}

int main() {
    //srand(time(NULL));
    srand(0);
    half h1, h2, h3, h4, d1fh, f1h, i1h, d1ih;
    uint16_t i1, i2, i3, i4, d1i;
	double d1, i1d, i2d, i3d, i4d;
    float d1f, i1hf, d1ihf;
    for (int x = 0; ; x++)
    {
        //bits_cast<uint16_t>(h2) = rand();
        //bits_cast<uint16_t>(h3) = rand();
        //bits_cast<uint16_t>(h4) = rand();
        //h1 = half_float::fma(h2, h3, h4);
        //h1f = float(h1);
        i1 = rand(); i2 = rand(); i3 = rand();
        d1 = double(bits_cast<half>(i2)) * double(bits_cast<half>(i3)) + double(bits_cast<half>(i4));
        d1f = float(d1);
        d1fh = half(d1f);
        d1i = float_downcast<DOUBLE,HALF>(bits_cast<uint64_t>(d1));
        i1 = float_fma<HALF>(bits_cast<uint16_t>(h2), bits_cast<uint16_t>(h3), bits_cast<uint16_t>(h4));
        i1h = bits_cast<half>(i1);
        i1hf = float(i1h);
        //d1h = half(d1);
        d1ih = bits_cast<half>(d1i);
        d1ihf = float(d1ih);
        if (std::isnan(float(d1ihf))) {
            assert(std::isnan(i1h));
            assert(std::isnan(i1hf));
        } else {
            //assert(bits_cast<uint16_t>(h1) == bits_cast<uint16_t>(f1h));
            //assert(i1 == bits_cast<uint16_t>(h1));
            //assert(i1 == bits_cast<uint16_t>(d1h));
            assert(i1 == d1i);
        }
    }
}
