from src.eval.metrics import idk_rate, is_idk_response


class TestIsIdkResponse:
    def test_recognizes_common_idk_phrasings(self):
        assert is_idk_response("I don't know based on the given context.")
        assert is_idk_response("I cannot find any mention of this in the repository.")

    def test_does_not_flag_a_confident_grounded_answer(self):
        assert not is_idk_response("AuthenticationMiddleware is defined in authentication.py.")


class TestIdkRate:
    def test_computes_the_fraction_of_idk_answers(self):
        answers = [
            "I don't know.",
            "AuthenticationMiddleware is defined in authentication.py.",
            "I don't know, this isn't in the repo.",
            "Router dispatches requests to routes.",
        ]

        assert idk_rate(answers) == 0.5

    def test_returns_zero_for_an_empty_list(self):
        assert idk_rate([]) == 0.0

    def test_returns_one_when_every_answer_is_idk(self):
        assert idk_rate(["I don't know.", "I do not know."]) == 1.0
