#ifndef MIDNIGHT_H
#define MIDNIGHT_H

#include "miner.h"

extern int midnight_test(unsigned char *pdata, const unsigned char *ptarget,
			uint32_t nonce);
extern void midnight_regenhash(struct work *work);

#endif /* MIDNIGHT_H */
