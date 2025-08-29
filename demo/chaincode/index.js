'use strict';

const { Contract } = require('fabric-contract-api');
const shim = require('fabric-shim');  // <-- required to start chaincode

class FedAvgContract extends Contract {

    async initLedger(ctx) {
        console.info('Ledger initialized');
        return;
    }

    async logEvent(ctx, eventId, payload) {
        const exists = await ctx.stub.getState(eventId);
        if (exists && exists.length > 0) {
            throw new Error(`Event ${eventId} already exists`);
        }
        await ctx.stub.putState(eventId, Buffer.from(payload));
        ctx.stub.setEvent('NewFedAvgEvent', Buffer.from(payload));
        return `Event ${eventId} logged`;
    }

    async queryEvent(ctx, eventId) {
        const eventJSON = await ctx.stub.getState(eventId);
        if (!eventJSON || eventJSON.length === 0) {
            throw new Error(`Event ${eventId} does not exist`);
        }
        return eventJSON.toString();
    }

    async queryAllEvents(ctx, startKey, endKey) {
        let results = [];
        for await (const { key, value } of ctx.stub.getStateByRange(startKey, endKey)) {
            results.push({ key, value: value.toString() });
        }
        return JSON.stringify(results);
    }
}

// Start the chaincode service
shim.start(new FedAvgContract());
