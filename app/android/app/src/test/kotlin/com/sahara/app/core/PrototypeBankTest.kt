package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PrototypeBankTest {
    @Test
    fun namesTheClosestCentroidAndRejectsFarVectors() {
        val bank = PrototypeBank(3)
        bank.add("Raj", floatArrayOf(1f, 0.1f, 0f))
        bank.add("Raj", floatArrayOf(0.9f, 0f, 0.1f))
        bank.add("Prachi", floatArrayOf(0f, 1f, 0f))
        assertEquals(2, bank.count("Raj"))
        assertEquals("Raj", bank.identify(floatArrayOf(2f, 0.1f, 0f), 0.7f)!!.name)
        assertEquals("Prachi", bank.identify(floatArrayOf(0.1f, 3f, 0f), 0.7f)!!.name)
        // Equally far from both: below the threshold, so "Someone".
        assertNull(bank.identify(floatArrayOf(0f, 0f, 1f), 0.7f)!!.name)
    }

    @Test
    fun emptyBankMatchesNobody() {
        assertNull(PrototypeBank(2).identify(floatArrayOf(1f, 0f), 0.5f))
    }
}
