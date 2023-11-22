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

	static constexpr Int EXP_ENCODING_BIAS = (BIT_LEASTSIG<<BITS_EXP) / 2 - 1;
	static constexpr UInt EXP_ENCODED_MAX = (BIT_LEASTSIG<<BITS_EXP) - 2;
	static constexpr UInt EXP_ENCODED_MIN = 1;
	static constexpr Int EXP_DECODED_MAX = EXP_ENCODED_MAX - EXP_ENCODING_BIAS;
	static constexpr Int EXP_DECODED_MIN = EXP_ENCODED_MIN - EXP_ENCODING_BIAS;
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
			(R==std::round_toward_infinity) ? (value+(~(value>>Float::POS_SIGN)&(guard|sticky))) :
			(R==std::round_toward_neg_infinity) ? (value+((value>>Float::POS_SIGN)&(guard|sticky))) :
			value;
}

template <class Float1, class Float=HALF, std::float_round_style R = std::round_to_nearest>
typename Float::UInt float_downcast(typename Float1::UInt bits1)
{
	constexpr auto const BITS_DIFF = Float1::BITS - Float::BITS;
	constexpr auto const BITS_M_DIFF = Float1::BITS_M - Float::BITS_M;
	constexpr auto const EXP_BIAS_DIFF = Float1::EXP_ENCODING_BIAS - Float::EXP_ENCODING_BIAS;
	//UInt1 bits1 = bits_cast<UInt1>(d);
	typename Float::UInt sign = static_cast<typename Float::UInt>(bits1>>BITS_DIFF) & Float::BIT_SIGN;
	bits1 &= Float1::MASK_ABS;
	if (bits1 > Float1::INF)
		// nan
		return sign | Float::INF | Float::BIT_M_MOSTSIG | ((bits1>>BITS_M_DIFF)&Float::MASK_M);
	else if (bits1 == Float1::INF)
		// inf
		return sign | Float::INF;
	else if (bits1 >= (Float1::EXP_ENCODING_BIAS + Float::EXP_DECODED_MAX + 1) << Float1::POS_EXP)
		// overflow -> infinity
			// it's notable here that summation is used instead of condition, for handling positive and negative extrema together
		return	(R==std::round_toward_infinity) ? (sign+Float::INF-(sign>>Float::POS_SIGN)) :
				(R==std::round_toward_neg_infinity) ? (sign+Float::INF-1+(sign>>Float::POS_SIGN)) :
				(R==std::round_toward_zero) ? (sign|(Float::INF-1)) :
				(sign|Float::INF);
	else if (bits1 >= (Float::EXP_DECODED_MIN + Float1::EXP_ENCODING_BIAS) << Float1::POS_EXP)
	//else if (bits1 >= (Float::EXP_BIAS_DIFF + 1) << Float1::POS_EXP)
		// normal value
		return float_round<Float,R>(
			sign|(((bits1>>Float1::POS_EXP)-EXP_BIAS_DIFF)<<Float::POS_EXP)|((bits1>>BITS_M_DIFF)&Float::MASK_M), 
			(bits1>>(BITS_M_DIFF-1))&1,
			(bits1&(Float1::MASK_M>>(Float::BITS_M+1)))!=0
		);
	else if (bits1 >= (EXP_BIAS_DIFF - Float::BITS_M) << Float1::POS_EXP) {
		// subnormal value
		int i = EXP_BIAS_DIFF + Float1::BITS_M - Float::BITS_M - (bits1 >> Float1::POS_EXP);
		bits1 = (bits1&Float1::MASK_M) | Float1::BIT_EXP_LEASTSIG;
		return float_round<Float,R>(
			sign | (bits1>>(i+1)),
			(bits1>>i)&1,
			(bits1&((typename Float1::UInt{1}<<i)-1))!=0
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
		typename Float::Int exp = -Float::EXP_ENCODING_BIAS;
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
			//return absz ? z : /*half(detail::binary, (half::round_style*/(R==std::round_toward_neg_infinity) ? (z|sign) : (z&sign)/*)*/;
			return  (absz) ? (z) :
					(R==std::round_toward_neg_infinity) ? (z|sign) :
					(R==std::round_toward_infinity) ? (z&sign) :
					sign;
					
		for(; absx<Float::BIT_EXP_LEASTSIG; absx<<=1,--exp) ;
		for(; absy<Float::BIT_EXP_LEASTSIG; absy<<=1,--exp) ;
		int sticky_bit = 0;
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
			expz += (absz >> Float::BITS_M);
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
			// the purpose of the |(..) is to ensure that half values with a little extra round upward?
			// the effect may be slightly different than correct
			//mz = (d<(Float::BITS_M*2+3)) ? ((mz>>d)|((mz&((typename Float::ULongInt{1}<<d)-1))!=0)) : 1;
			if (d < Float::BITS_M*2+3) {
				sticky_bit = ((mz&((typename Float::ULongInt{1}<<d)-1))!=0);
				mz >>= d;
			} else {
				sticky_bit = 1;
				mz = 0;
			}
			//sticky_bit = (d<(Float::BITS_M*2+3)) ? ((mz&((typename Float::ULongInt{1}<<d)-1))!=0) : 1;
			//mz >>= d; expz = exp;
			expz = exp;
			//m <<= d;
			//exp = expz;
			if(sub)
			{
				m = m - mz;
				if(!m)
					return /*half(detail::binary, */typename Float::UInt{/*half::round_style*/R==std::round_toward_neg_infinity}<<Float::POS_SIGN/*)*/;
				for(; m<typename Float::UInt{1}<<(Float::BITS_M*2+3); m<<=1,--exp) ;
			}
			else
			{
				m += mz;
				i = m >> (Float::BITS_M*2+4);
				m = (m>>i) | (m&i); // this should probably be changed to put m&i in sticky bit.
				exp += i;
			}
		}
		if(exp > Float::EXP_ENCODED_MAX)
			//return half(detail::binary, detail::overflow<half::round_style>(sign));
			return	(R==std::round_toward_infinity) ? (sign+Float::INF-(sign>>Float::POS_SIGN)) :
					(R==std::round_toward_neg_infinity) ? (sign+Float::INF-1+(sign>>Float::POS_SIGN)) :
					(R==std::round_toward_zero) ? (sign|(Float::INF-1)) :
					(sign|Float::INF);
		else if(exp + Float::BITS_M < 0)
		//	return half(detail::binary, detail::underflow<half::round_style>(sign));
			return	(R==std::round_toward_infinity) ? (sign+1-(sign>>Float::POS_SIGN)) :
					(R==std::round_toward_neg_infinity) ? (sign+(sign>>Float::POS_SIGN)) :
					sign;
		//return half(detail::binary, detail::fixed2half<half::round_style,23,false,false,false>(m, exp-1, sign));
		//std::float_round_style R = half::round_style;
		constexpr auto F = Float::BITS_M*2+3;
		//bool S = false;
		//bool N = false;
		//bool I = false;
		exp -= 1;
		//int s = 0;
			/*if(S)
			{
				uint32 msign = sign_mask(m);
				m = (m^msign) - msign;
				sign = msign & 0x8000;
			}*/
			//unsigned int value;
			//int guard_bit;
			//int sticky_bit;
			/*if(N)
				for(; m<(static_cast<uint32>(1)<<F) && exp; m<<=1,--exp) ;
			else*/
			if(exp >= 0) {
				return float_round<Float,R>(
					sign+(exp<<Float::POS_EXP)+(m>>(F-Float::BITS_M)),
					(m>>(F-Float::BITS_M-1))&1,
					sticky_bit|((m&((Float::BIT_LEASTSIG<<(F-Float::BITS_M-1))-1))!=0)
				);
			} else if((unsigned int)exp + Float::BITS_M + (unsigned int)sizeof(m)*8 > F) {
				// it seems this should just be an 'else'
				// but i was running into the >> operator rolling
				// around instead of shifting, so added the check around
				// this. maybe the shift operand was getting modulod.
				// cpu intel core i7-7500U
				// g++ 9.4.1 20211121
				// date 2023-11-21
				return float_round<Float,R>(
					sign+(m>>(F-Float::BITS_M-exp)),
					(m>>(F-Float::BITS_M-1-exp))&1,
					sticky_bit|((m&((Float::BIT_LEASTSIG<<(F-Float::BITS_M-1-exp))-1))!=0)
				);
			} else {
				return sign;
			}
			//return	(R==std::round_to_nearest) ? (value+(guard_bit&(sticky_bit|value))) :
			//		(R==std::round_toward_infinity) ? (value+(~(value>>Float::POS_SIGN)&(guard_bit|sticky_bit))) :
			//		(R==std::round_toward_neg_infinity) ? (value+((value>>Float::POS_SIGN)&(guard_bit|sticky_bit))) :
			//		value;
}

int main() {
	//srand(time(NULL));
	srand(0);
	half h1, h2, h3, h4, h5, d1fh, d5fh, f1h, i1h, d1ih, f5h, i5h, d5ih;
	uint16_t i1, i2, i3, i4, i5, d1i, d5i;
	float f2, f3, f4, f5;
	double d1, d2, d3, d4, d5, i1d, i2d, i3d, i4d, i5d;
	float d1f, d5f, i1hf, i5hf, d1ihf, d1fhf, d5ihf, d5fhf;
	for (int x = 0; ; x++)
	{
		//bits_cast<uint16_t>(h2) = rand();
		//bits_cast<uint16_t>(h3) = rand();
		//bits_cast<uint16_t>(h4) = rand();
		//h1 = half_float::fma(h2, h3, h4);
		//h1f = float(h1);
		i2 = rand(); i3 = rand(); i4 = rand();
		//if (x < 6) continue;
		h2 = bits_cast<half>(i2); h3 = bits_cast<half>(i3); h4 = bits_cast<half>(i4);
		f2 = float(h2); f3 = float(h3); f4 = float(h4);
		d2 = double(h2); d3 = double(h3); d4 = double(h4);
		d5 = d2 * d3;
		d5f = float(d5);
		d5fh = half(d5f);
		d5fhf = float(d5fh);
		uint64_t d5_i = bits_cast<uint64_t>(d5);
		d5i = float_downcast<DOUBLE,HALF>(d5_i);
		i5 = float_fma<HALF>(i2, i3, bits_cast<uint64_t>(half(0.0f)));
		i5h = bits_cast<half>(i5);
		i5hf = float(i5h);
		//d5h = half(d5);
		d5ih = bits_cast<half>(d5i);
		d5ihf = float(d5ih);
		d1 = d5 + d4;
		d1f = float(d1);
		d1fh = half(d1f);
		d1fhf = float(d1fh);
		uint64_t d1_i = bits_cast<uint64_t>(d1);
		d1i = float_downcast<DOUBLE,HALF>(d1_i);
		i1 = float_fma<HALF>(i2, i3, i4);
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
			assert(i5 == d5i);
			assert(i1 == d1i);
		}
	}
}
