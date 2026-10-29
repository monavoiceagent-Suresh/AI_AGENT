-- The number an inbound call was routed by (digits, country code first). It can differ from `to_number`, which is
-- exactly what Exotel put in the stream: on a flow whose `To` is not the dialled ExoPhone, the route is found by the
-- ExoPhone from Exotel's call record. Reports to the dashboard carry this number.
ALTER TABLE calls ADD COLUMN dialled_number text;
